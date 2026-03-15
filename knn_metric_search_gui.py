#!/usr/bin/env python3
"""
Tkinter GUI for k-NN hyperparameter search.
"""
import os
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Import metric names from the main script
from distance_metrics import METRIC_NAMES


def get_script_path():
    """Return path to knn_metric_search.py in the same directory as this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "knn_metric_search.py")


class KnnSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("k-NN Hyperparameter Search")
        self.root.minsize(600, 500)
        w, h = 750, 600
        self.root.geometry(f"{w}x{h}")
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")

        self.train_path = tk.StringVar()
        self.test_path = tk.StringVar()
        self.k_min = tk.StringVar(value="1")
        self.k_max = tk.StringVar(value="10")
        self.metric_vars = {m: tk.BooleanVar(value=True) for m in METRIC_NAMES}
        self.running = False
        self.process = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Datasets section ---
        ds_frame = ttk.LabelFrame(main, text="Datasets", padding=5)
        ds_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(ds_frame, text="Training:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        train_entry = ttk.Entry(ds_frame, textvariable=self.train_path, width=50)
        train_entry.grid(row=0, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Button(ds_frame, text="Browse...", command=self._browse_train).grid(
            row=0, column=2, padx=2, pady=2
        )

        ttk.Label(ds_frame, text="Test:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        test_entry = ttk.Entry(ds_frame, textvariable=self.test_path, width=50)
        test_entry.grid(row=1, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Button(ds_frame, text="Browse...", command=self._browse_test).grid(
            row=1, column=2, padx=2, pady=2
        )

        ds_frame.columnconfigure(1, weight=1)

        # --- K range section ---
        k_frame = ttk.LabelFrame(main, text="k range", padding=5)
        k_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(k_frame, text="Min k:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        ttk.Spinbox(k_frame, textvariable=self.k_min, from_=1, to=9999, width=8).grid(
            row=0, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(k_frame, text="Max k:").grid(row=0, column=2, sticky=tk.W, padx=(15, 5), pady=2)
        ttk.Spinbox(k_frame, textvariable=self.k_max, from_=1, to=9999, width=8).grid(
            row=0, column=3, sticky=tk.W, padx=2, pady=2
        )

        # --- Metrics section ---
        metrics_frame = ttk.LabelFrame(main, text="Distance metrics", padding=5)
        metrics_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(metrics_frame, text="Select all", command=self._select_all_metrics).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(metrics_frame, text="Deselect all", command=self._deselect_all_metrics).pack(
            side=tk.LEFT, padx=(0, 5))

        metrics_inner = ttk.Frame(metrics_frame)
        metrics_inner.pack(fill=tk.X, pady=(5, 0))
        for i, metric in enumerate(METRIC_NAMES):
            cb = ttk.Checkbutton(
                metrics_inner, text=metric, variable=self.metric_vars[metric]
            )
            cb.grid(row=i // 4, column=i % 4, sticky=tk.W, padx=(0, 15), pady=1)

        # --- Status / output ---
        out_frame = ttk.LabelFrame(main, text="Status", padding=5)
        out_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.output_text = scrolledtext.ScrolledText(
            out_frame, height=12, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # --- Run button ---
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        self.run_btn = ttk.Button(btn_frame, text="Run search", command=self._run_search)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 5))

    def _browse_train(self):
        path = filedialog.askopenfilename(
            title="Select training CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.train_path.set(path)

    def _browse_test(self):
        path = filedialog.askopenfilename(
            title="Select test CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.test_path.set(path)

    def _select_all_metrics(self):
        for v in self.metric_vars.values():
            v.set(True)

    def _deselect_all_metrics(self):
        for v in self.metric_vars.values():
            v.set(False)

    def _log(self, msg):
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, msg)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    def _run_search(self):
        if self.running:
            messagebox.showwarning("Search running", "A search is already in progress.")
            return

        train = self.train_path.get().strip()
        test = self.test_path.get().strip()
        if not train:
            messagebox.showerror("Error", "Please select a training dataset.")
            return
        if not test:
            messagebox.showerror("Error", "Please select a test dataset.")
            return

        try:
            k_min = int(self.k_min.get())
            k_max = int(self.k_max.get())
            if k_min < 1 or k_max < k_min:
                raise ValueError("Invalid k range: min must be >= 1 and max >= min.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid k range: {e}")
            return

        selected = [m for m, v in self.metric_vars.items() if v.get()]
        if not selected:
            messagebox.showerror("Error", "Please select at least one distance metric.")
            return

        k_values = ",".join(str(k) for k in range(k_min, k_max + 1))
        metrics_str = ",".join(selected)

        script_path = get_script_path()
        cmd = [
            sys.executable,
            script_path,
            "--data", train,
            "--test-data", test,
            "--k-values", k_values,
            "--metrics", metrics_str,
            "--log-each-test",
        ]

        self.running = True
        self.run_btn.configure(state=tk.DISABLED)
        self._log("Starting search...\n")
        self._log(f"Command: {' '.join(cmd)}\n\n")

        def run():
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                popen_kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,
                    "env": env,
                }
                if sys.platform != "win32":
                    popen_kwargs["start_new_session"] = True
                self.process = subprocess.Popen(cmd, **popen_kwargs)
                for line in self.process.stdout:
                    self.root.after(0, lambda l=line: self._log(l))
                self.process.wait()
                self.root.after(0, lambda: self._on_search_done(None))
            except Exception as e:
                self.root.after(0, lambda: self._on_search_done(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _on_search_done(self, error):
        self.running = False
        self.process = None
        self.run_btn.configure(state=tk.NORMAL)
        if error:
            self._log(f"\nError: {error}\n")
            messagebox.showerror("Error", error)
        else:
            self._log("\nSearch completed.\n")

    def _on_closing(self):
        """Kill the search subprocess if running, then close the window."""
        if self.running and self.process is not None:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=3)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                try:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                except (ProcessLookupError, OSError):
                    pass
            self.process = None
            self.running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    app = KnnSearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
