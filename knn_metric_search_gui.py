#!/usr/bin/env python3
"""
Tkinter GUI for k-NN hyperparameter search.
"""
import os
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from knn_tuner import METRIC_NAMES

# Regex to parse per-test output: [Phase 1] k=1, metric=euclidean, accuracy=0.8523, time=12.34s
RESULT_LINE_RE = re.compile(
    r"\[.*?\]\s*k=(\d+),\s*metric=([^,]+),\s*accuracy=([\d.]+),\s*time=([\d.]+)s"
)
# Regex for final results block: k=  1  metric=euclidean  accuracy=0.8523  time=12.34s
RESULT_FINAL_RE = re.compile(
    r"k=\s*(\d+)\s+metric=(\S+)\s+accuracy=([\d.]+)\s+time=([\d.]+)s"
)


class KnnSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("k-NN Hyperparameter Search")
        self.root.resizable(True, True)
        # Default 480x800 if screen fits, else smaller
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w = min(520, screen_w)
        h = min(900, screen_h)
        self.root.minsize(400, 500)
        self.root.geometry(f"{w}x{h}")
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2
        self.root.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")

        self.train_path = tk.StringVar()
        self.test_path = tk.StringVar()
        self.k_min = tk.StringVar(value="1")
        self.k_max = tk.StringVar(value="2")  # set to dimensionality when data loaded
        self.metric_vars = {m: tk.BooleanVar(value=True) for m in METRIC_NAMES}
        self.running = False
        self.process = None
        self.all_results = []  # (k, metric, acc, time)
        self.best_so_far = None  # (k, metric, acc, time) or None
        self.total_tests = 0  # set when search starts
        self.max_k_allowed = None  # min(n_features, n_samples) from data; None until loaded
        self._in_k_validation = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _build_ui(self):
        # Scrollable container so all components visible when window is small
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        main = ttk.Frame(canvas, padding=10)

        _scroll_initialized = [False]
        def _update_scroll_region(_=None):
            main.update_idletasks()
            try:
                b = canvas.bbox("all")
                if b and (b[2] - b[0]) > 10 and (b[3] - b[1]) > 10:
                    canvas.configure(scrollregion=b)
                else:
                    w = main.winfo_reqwidth()
                    h = main.winfo_reqheight()
                    canvas.configure(scrollregion=(0, 0, max(w, 400), max(h, 700)))
                if not _scroll_initialized[0]:
                    canvas.yview_moveto(0)
                    _scroll_initialized[0] = True
            except Exception:
                canvas.configure(scrollregion=(0, 0, 600, 1200))
                if not _scroll_initialized[0]:
                    canvas.yview_moveto(0)
                    _scroll_initialized[0] = True
        main.bind("<Configure>", _update_scroll_region)
        canvas_window = canvas.create_window((0, 0), window=main, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=max(event.width, 400))
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if hasattr(event, "delta"):
                d = int(-event.delta) if sys.platform == "darwin" else int(-event.delta / 120)
                canvas.yview_scroll(d, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            elif getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
        canvas.bind("<Enter>", lambda _: (
            canvas.bind_all("<MouseWheel>", _on_mousewheel),
            canvas.bind_all("<Button-4>", _on_mousewheel),
            canvas.bind_all("<Button-5>", _on_mousewheel),
        ))
        canvas.bind("<Leave>", lambda _: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Button-4>"),
            canvas.unbind_all("<Button-5>"),
        ))

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main.columnconfigure(0, weight=1)
        row = 0

        # --- Datasets section ---
        ds_frame = ttk.LabelFrame(main, text="Datasets", padding=5)
        ds_frame.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        ds_frame.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(ds_frame, text="Training:").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        train_entry = ttk.Entry(ds_frame, textvariable=self.train_path)
        train_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(ds_frame, text="Browse...", command=self._browse_train).grid(row=0, column=2, padx=2, pady=2)
        ttk.Label(ds_frame, text="Test:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=2)
        test_entry = ttk.Entry(ds_frame, textvariable=self.test_path)
        test_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(ds_frame, text="Browse...", command=self._browse_test).grid(row=1, column=2, padx=2, pady=2)

        # --- K range section ---
        k_frame = ttk.LabelFrame(main, text="k range (1 ≤ k_low < k_high ≤ dimensionality)", padding=5)
        k_frame.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        k_frame.columnconfigure(4, weight=1)
        row += 1
        ttk.Label(k_frame, text="Min k:").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        self.k_min_spin = ttk.Spinbox(k_frame, textvariable=self.k_min, from_=1, to=9999, width=6)
        self.k_min_spin.grid(row=0, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(k_frame, text="Max k:").grid(row=0, column=2, sticky="w", padx=(15, 5), pady=2)
        self.k_max_spin = ttk.Spinbox(k_frame, textvariable=self.k_max, from_=1, to=9999, width=6)
        self.k_max_spin.grid(row=0, column=3, sticky="w", padx=2, pady=2)
        self.k_limit_label = ttk.Label(k_frame, text="(max k from data: —)", foreground="gray")
        self.k_limit_label.grid(row=0, column=4, sticky="w", padx=(10, 0), pady=2)

        self.k_min.trace_add("write", lambda *a: self._on_k_changed("min"))
        self.k_max.trace_add("write", lambda *a: self._on_k_changed("max"))

        # --- Metrics section (responsive grid: 2–4 cols based on width) ---
        metrics_frame = ttk.LabelFrame(main, text="Distance metrics", padding=5)
        metrics_frame.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        metrics_frame.columnconfigure(0, weight=1)
        row += 1
        btn_row = ttk.Frame(metrics_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Select all", command=self._select_all_metrics).pack(side="left", padx=(0, 5))
        ttk.Button(btn_row, text="Deselect all", command=self._deselect_all_metrics).pack(side="left", padx=(0, 5))
        self.metrics_inner = ttk.Frame(metrics_frame)
        self.metrics_inner.pack(fill="x", pady=(5, 0))
        self.metric_checkboxes = []
        for metric in METRIC_NAMES:
            cb = ttk.Checkbutton(self.metrics_inner, text=metric, variable=self.metric_vars[metric])
            self.metric_checkboxes.append(cb)
        self._layout_metrics_grid()
        metrics_frame.bind("<Configure>", lambda e: self._layout_metrics_grid(e.width))

        # --- Results panel ---
        results_container = tk.Frame(main, bg="#e8f4f8", relief=tk.RIDGE, bd=2)
        results_container.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        results_container.columnconfigure(0, weight=1)
        row += 1
        results_header = tk.Label(results_container, text=" RESULTS ", font=("", 10, "bold"),
            bg="#2c5aa0", fg="white", padx=8, pady=4)
        results_header.pack(fill="x")
        results_inner = tk.Frame(results_container, bg="#e8f4f8", padx=8, pady=6)
        results_inner.pack(fill="x")
        self.best_label = tk.Label(results_inner, text="Best so far: —", font=("", 9, "bold"),
            bg="#e8f4f8", fg="#1a4d6e")
        self.best_label.pack(anchor="w")
        ttk.Label(results_inner, text="Top 1%:").pack(anchor="w")
        self.top1pct_text = scrolledtext.ScrolledText(
            results_inner, height=3, wrap=tk.WORD, state=tk.DISABLED,
            font=("Courier", 8), bg="#f0f8ff", fg="#333"
        )
        self.top1pct_text.pack(fill="x", pady=(2, 0))

        # --- Progress display ---
        progress_container = tk.Frame(main, bg="#f5f0e6", relief=tk.GROOVE, bd=2)
        progress_container.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        progress_container.columnconfigure(0, weight=1)
        row += 1
        progress_header = tk.Label(progress_container, text=" PROGRESS ", font=("", 10, "bold"),
            bg="#8b7355", fg="white", padx=8, pady=4)
        progress_header.pack(fill="x")
        progress_inner = tk.Frame(progress_container, bg="#f5f0e6", padx=8, pady=6)
        progress_inner.pack(fill="x")
        self.avg_time_label = tk.Label(progress_inner, text="Avg time per test: —", font=("", 9),
            bg="#f5f0e6", fg="#4a3728")
        self.avg_time_label.pack(anchor="w")
        self.tests_progress_label = tk.Label(progress_inner,
            text="Tests: — completed / — remaining — —% complete", font=("", 9),
            bg="#f5f0e6", fg="#4a3728")
        self.tests_progress_label.pack(anchor="w")

        # --- Status / output (expandable) ---
        out_frame = ttk.LabelFrame(main, text="Status", padding=5)
        out_frame.grid(row=row, column=0, sticky="nsew", pady=(0, 5))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)
        row += 1
        self.output_text = scrolledtext.ScrolledText(
            out_frame, height=8, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 9)
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")

        # --- Run button ---
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=row, column=0, sticky="ew", pady=(5, 0))
        btn_frame.columnconfigure(0, weight=1)
        self.run_btn = ttk.Button(btn_frame, text="Run search", command=self._run_search)
        self.run_btn.pack(side="left", padx=(0, 5))

        # Row weights: Status gets all extra space
        main.rowconfigure(row - 1, weight=1)
        # Force layout so scroll region and all components are visible on start
        main.update_idletasks()
        self.root.update_idletasks()
        self.root.update()
        _update_scroll_region()
        for delay in (0, 50, 150, 300):
            self.root.after(delay, _update_scroll_region)

    def _layout_metrics_grid(self, width=None):
        """Responsive grid: 2–4 columns based on available width."""
        w = width if width is not None else self.metrics_inner.winfo_width()
        if w < 1:
            w = self.metrics_inner.winfo_reqwidth() or 400
        if w < 350:
            ncols = 2
        elif w < 500:
            ncols = 3
        else:
            ncols = 4
        for i, cb in enumerate(self.metric_checkboxes):
            cb.grid(row=i // ncols, column=i % ncols, sticky="w", padx=(0, 12), pady=1)

    def _load_max_k_from_data(self):
        """Load training CSV to get max k = min(n_features, n_samples)."""
        train = self.train_path.get().strip()
        if not train or not os.path.isfile(train):
            self.max_k_allowed = None
            self.k_limit_label.configure(text="(max k from data: —)")
            self.k_min_spin.configure(to=9999)
            self.k_max_spin.configure(to=9999)
            return
        try:
            import pandas as pd
            df = pd.read_csv(train)
            class_cols = [c for c in df.columns if c.lower() == "class"]
            n_features = len(df.columns) - (1 if class_cols else 0)
            n_samples = len(df)
            self.max_k_allowed = min(n_features, n_samples)
            self.k_limit_label.configure(text=f"(max k from data: {self.max_k_allowed})")
            self.k_min_spin.configure(to=max(1, self.max_k_allowed - 1))
            self.k_max_spin.configure(to=self.max_k_allowed)
            # Default to full range 1 to dimensionality; user can constrain
            self.k_min.set("1")
            self.k_max.set(str(self.max_k_allowed))
            self._enforce_k_constraints()
        except Exception:
            self.max_k_allowed = None
            self.k_limit_label.configure(text="(max k from data: —)")

    def _enforce_k_constraints(self):
        """Enforce 1 <= k_low < k_high <= max_k_allowed."""
        try:
            k_min_val = int(self.k_min.get())
        except ValueError:
            self.k_min.set("1")
            return
        try:
            k_max_val = int(self.k_max.get())
        except ValueError:
            n = self.max_k_allowed if self.max_k_allowed is not None else 2
            self.k_max.set(str(n))
            return

        n = self.max_k_allowed if self.max_k_allowed is not None else 9999
        k_min_val = max(1, min(k_min_val, n - 1))  # k_low < n so k_high can be n
        k_max_val = min(max(k_min_val + 1, k_max_val), n)  # k_high > k_low and k_high <= n
        if k_max_val <= k_min_val:
            k_max_val = min(k_min_val + 1, n)
        self.k_min.set(str(k_min_val))
        self.k_max.set(str(k_max_val))

    def _on_k_changed(self, which):
        """Validate k on change; enforce 1 <= k_low < k_high <= n."""
        if self._in_k_validation:
            return
        self.root.after_idle(self._do_k_validation)

    def _do_k_validation(self):
        self._in_k_validation = True
        try:
            self._enforce_k_constraints()
        finally:
            self._in_k_validation = False

    def _browse_train(self):
        path = filedialog.askopenfilename(
            title="Select training CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.train_path.set(path)
            self._load_max_k_from_data()

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

    def _process_and_log(self, line):
        """Parse line for result data; if found, update results panel. Then log the line."""
        for regex in (RESULT_LINE_RE, RESULT_FINAL_RE):
            m = regex.search(line)
            if m:
                k = int(m.group(1))
                metric = m.group(2).strip()
                acc = float(m.group(3))
                elapsed = float(m.group(4))
                self.root.after(0, lambda k=k, m=metric, a=acc, t=elapsed: self._on_new_result(k, m, a, t))
                break
        self._log(line)

    def _on_new_result(self, k, metric, acc, elapsed):
        """Add result (dedupe by k,metric) and refresh the results panel."""
        seen = {(r[0], r[1]) for r in self.all_results}
        if (k, metric) in seen:
            return
        self.all_results.append((k, metric, acc, elapsed))
        valid = [r for r in self.all_results if r[2] == r[2]]  # exclude nan
        if valid:
            self.best_so_far = max(valid, key=lambda x: x[2])
        self._update_results_panel()

    def _update_results_panel(self):
        """Refresh best-so-far label, top 1% table, and progress display."""
        if self.best_so_far:
            k, m, a, t = self.best_so_far
            self.best_label.configure(text=f"Best so far: k={k}, metric={m}, accuracy={a:.4f}")
        else:
            self.best_label.configure(text="Best so far: —")

        valid = [r for r in self.all_results if r[2] == r[2]]
        valid_sorted = sorted(valid, key=lambda x: x[2], reverse=True)
        n_top = max(1, int(len(valid_sorted) * 0.01))
        top1pct = valid_sorted[:n_top]

        self.top1pct_text.configure(state=tk.NORMAL)
        self.top1pct_text.delete(1.0, tk.END)
        for k, m, a, t in top1pct:
            self.top1pct_text.insert(tk.END, f"k={k:3d}  {m:22s}  {a:.4f}\n")
        self.top1pct_text.configure(state=tk.DISABLED)

        # Progress: avg time, completed/remaining, %
        completed = len(self.all_results)
        if self.total_tests > 0:
            remaining = self.total_tests - completed
            pct = 100.0 * completed / self.total_tests
            self.tests_progress_label.configure(
                text=f"Tests: {completed} completed / {remaining} remaining — {pct:.1f}% complete"
            )
        else:
            self.tests_progress_label.configure(text="Tests: — completed / — remaining — —% complete")

        times = [r[3] for r in self.all_results if r[3] == r[3]]  # exclude nan
        if times:
            avg = sum(times) / len(times)
            self.avg_time_label.configure(text=f"Avg time per test: {avg:.2f}s")
        else:
            self.avg_time_label.configure(text="Avg time per test: —")

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

        self._load_max_k_from_data()
        self._enforce_k_constraints()
        try:
            k_min = int(self.k_min.get())
            k_max = int(self.k_max.get())
            if k_min < 1:
                raise ValueError("k_min must be ≥ 1.")
            if k_max <= k_min:
                raise ValueError("k_max must be > k_min.")
            if self.max_k_allowed is not None and k_max > self.max_k_allowed:
                raise ValueError(f"k_max must be ≤ {self.max_k_allowed} (data dimensionality).")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        selected = [m for m, v in self.metric_vars.items() if v.get()]
        if not selected:
            messagebox.showerror("Error", "Please select at least one distance metric.")
            return

        k_values = ",".join(str(k) for k in range(k_min, k_max + 1))
        metrics_str = ",".join(selected)

        cmd = [
            sys.executable,
            "-m", "knn_tuner.search",
            "--data", train,
            "--test-data", test,
            "--k-values", k_values,
            "--metrics", metrics_str,
            "--log-each-test",
        ]

        self.running = True
        self.run_btn.configure(state=tk.DISABLED)
        self.all_results = []
        self.best_so_far = None
        n_k = k_max - k_min + 1
        n_metrics = len(selected)
        self.total_tests = n_k * n_metrics
        self._update_results_panel()
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
                    self.root.after(0, lambda l=line: self._process_and_log(l))
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
            for ext in ("xlsx", "csv"):
                path = os.path.abspath(os.path.join("results", f"knn_metric_search_results.{ext}"))
                if os.path.exists(path):
                    messagebox.showinfo("Search complete", f"Results saved to:\n{path}")
                    break

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
