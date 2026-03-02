import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os

from detectors import (
    detect_hough,
    detect_min_enclosing,
    detect_canny_hough,
    detect_multi_scale_fusion
)

try:
    from eval_metrics import evaluate_circle
except ImportError:
    def evaluate_circle(gray, x, y, r):
        return 0.0, 0.0

METHODS = {
    "Hough Circle Transform": {
        "func": detect_hough,
        "params": ["dp", "minDist", "param1", "param2", "minRadius", "maxRadius"]
    },
    "MinEnclosing": {
        "func": detect_min_enclosing,
        "params": ["binary_thresh", "min_area"]
    },
    "Canny+Hough": {
        "func": detect_canny_hough,
        "params": ["canny_low", "canny_high", "dp", "minDist", "param2", "minRadius", "maxRadius"]
    },
    "Multi-scale Fusion": {
        "func": detect_multi_scale_fusion,
        "params": ["fusion_thresh", "dp", "minDist", "param2", "minRadius", "maxRadius"]
    }
}

PARAM_DEF = {
    "dp": {"label": "Hough dp", "type": "float", "range": (1.0, 3.0), "default": 1.2, "step": 0.1},
    "minDist": {"label": "Min Dist", "type": "int", "range": (1, 500), "default": 30},
    "param1": {"label": "Hough P1 (Gradient)", "type": "int", "range": (1, 300), "default": 50},
    "param2": {"label": "Hough P2 (Accum)", "type": "int", "range": (1, 200), "default": 30},
    "minRadius": {"label": "Min Radius", "type": "int", "range": (1, 200), "default": 10},
    "maxRadius": {"label": "Max Radius", "type": "int", "range": (10, 500), "default": 100},
    "binary_thresh": {"label": "Bin Thresh (0=Otsu)", "type": "int", "range": (0, 255), "default": 0},
    "min_area": {"label": "Min Area", "type": "int", "range": (10, 10000), "default": 100},
    "canny_low": {"label": "Canny Low", "type": "int", "range": (0, 255), "default": 50},
    "canny_high": {"label": "Canny High", "type": "int", "range": (0, 255), "default": 150},
    "fusion_thresh": {"label": "Fusion Threshold", "type": "float", "range": (0.1, 0.9), "default": 0.3, "step": 0.05}
}

class CircleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drilling-Hole Circle-Detection App")
        self.root.geometry("1400x800")
        
        self.original_image_pil = None
        self.grey_np = None
        self.color_np = None
        
        self.left_img_tk = None
        self.result_images_tk = [] 
        self.export_image_pil = None
        
        self.param_vars = {}
        for k, v in PARAM_DEF.items():
            if v["type"] == "int":
                self.param_vars[k] = tk.IntVar(value=v["default"])
            else:
                self.param_vars[k] = tk.DoubleVar(value=v["default"])
                
        self.setup_ui()
        
    def setup_ui(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT PANE ---
        self.left_frame = ttk.LabelFrame(main_pane, text="Input Image")
        main_pane.add(self.left_frame, minsize=400)
        
        self.btn_load = ttk.Button(self.left_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(pady=5)
        
        self.left_label = ttk.Label(self.left_frame, text="No Image Loaded", anchor=tk.CENTER)
        self.left_label.pack(fill=tk.BOTH, expand=True)

        # --- MIDDLE PANE ---
        self.mid_frame = ttk.LabelFrame(main_pane, text="Controls")
        main_pane.add(self.mid_frame, minsize=350)
        
        ttk.Label(self.mid_frame, text="Detection Method:").pack(pady=(10,0))
        self.method_var = tk.StringVar(value=list(METHODS.keys())[0])
        self.dropdown = ttk.Combobox(self.mid_frame, textvariable=self.method_var, values=list(METHODS.keys()), state="readonly")
        self.dropdown.pack(fill=tk.X, padx=10, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_method_changed)
        
        self.params_frame = ttk.Frame(self.mid_frame)
        self.params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.on_method_changed()
        
        ttk.Separator(self.mid_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.compare_var = tk.BooleanVar(value=False)
        self.chk_compare = ttk.Checkbutton(self.mid_frame, text="Enable Compare Mode", variable=self.compare_var)
        self.chk_compare.pack(pady=5)
        
        self.btn_run = ttk.Button(self.mid_frame, text="Run Detection", command=self.run_detection)
        self.btn_run.pack(pady=10, fill=tk.X, padx=10)
        
        self.btn_export = ttk.Button(self.mid_frame, text="Export Results (PNG)", command=self.export_results, state=tk.DISABLED)
        self.btn_export.pack(pady=(0,10), fill=tk.X, padx=10)
        
        ttk.Separator(self.mid_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        btn_frame = ttk.Frame(self.mid_frame)
        btn_frame.pack(fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(btn_frame, text="Load Config", command=self.load_config).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2,0))

        # --- RIGHT PANE ---
        self.right_frame = ttk.LabelFrame(main_pane, text="Results")
        main_pane.add(self.right_frame, minsize=500)
        
        self.right_single_label = ttk.Label(self.right_frame, text="Result will appear here", anchor=tk.CENTER)
        self.right_single_label.pack(fill=tk.BOTH, expand=True)
        
        self.right_grid_frame = ttk.Frame(self.right_frame)
        self.grid_labels = []
        for i in range(4):
            lbl = ttk.Label(self.right_grid_frame, text=f"Tile {i}", anchor=tk.CENTER, relief=tk.SUNKEN)
            lbl.grid(row=i//2, column=i%2, sticky="nsew", padx=2, pady=2)
            self.grid_labels.append(lbl)
        self.right_grid_frame.columnconfigure(0, weight=1)
        self.right_grid_frame.columnconfigure(1, weight=1)
        self.right_grid_frame.rowconfigure(0, weight=1)
        self.right_grid_frame.rowconfigure(1, weight=1)
        
        # --- BOTTOM PANE ---
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_method_changed(self, event=None):
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        method = self.method_var.get()
        param_keys = METHODS[method]["params"]
        
        for k in param_keys:
            pdef = PARAM_DEF[k]
            frame = ttk.Frame(self.params_frame)
            frame.pack(fill=tk.X, pady=2)
            
            lbl = ttk.Label(frame, text=pdef["label"], width=20)
            lbl.pack(side=tk.LEFT)
            
            val_lbl = ttk.Label(frame, width=5)
            val_lbl.pack(side=tk.RIGHT)
            
            def update_lbl(*args, var=self.param_vars[k], l=val_lbl, is_int=(pdef["type"]=="int")):
                try:
                    val = var.get()
                    if is_int:
                        l.config(text=str(int(val)))
                    else:
                        l.config(text=f"{val:.2f}")
                except:
                    pass
                    
            self.param_vars[k].trace_add("write", update_lbl)
            update_lbl()
            
            slider = tk.Scale(frame, from_=pdef["range"][0], to=pdef["range"][1],
                              variable=self.param_vars[k], orient=tk.HORIZONTAL, 
                              resolution=pdef.get("step", 1), showvalue=0)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not path: return
        
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
            
        self.color_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Pillow expects RGB
        self.grey_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        self.original_image_pil = Image.fromarray(self.color_np)
        self.display_image(self.original_image_pil, self.left_label, max_size=(600, 600))
        self.status_var.set(f"Loaded: {os.path.basename(path)} | Size: {self.color_np.shape[1]}x{self.color_np.shape[0]}")
        self.btn_export.config(state=tk.DISABLED)
        
    def display_image(self, pil_img, label_widget, max_size=(600, 600)):
        img_copy = pil_img.copy()
        img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_copy)
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img 
        
    def _draw_result(self, cv_rgb, result, method_name=""):
        res_img = cv_rgb.copy()
        text = ""
        if result['success']:
            x, y = result['center']
            r = result['radius']
            # Draw circle boundary (Red)
            cv2.circle(res_img, (x, y), r, (255, 0, 0), max(2, int(r * 0.05)))
            # Draw center cross (Green)
            cv2.drawMarker(res_img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, max(10, int(r * 0.2)), 2)
            
            cov, conf = evaluate_circle(self.grey_np, x, y, r)
            text = f"{method_name} - R:{r}  Cov:{cov:.2f} Conf:{conf:.1f}%"
            bg_color, text_color = (0, 0, 0), (0, 255, 0)
        else:
            text = f"{method_name} - Failed"
            bg_color, text_color = (0, 0, 0), (255, 0, 0)
            
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(res_img, (5, 5), (5 + tw + 10, 5 + th + 10), bg_color, -1)
        cv2.putText(res_img, text, (10, 20 + th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return res_img, text

    def run_detection(self):
        if self.grey_np is None:
            messagebox.showwarning("Warning", "Please load image first!")
            return
            
        self.root.config(cursor="watch")
        self.root.update()
        
        try:
            if not self.compare_var.get():
                self.right_grid_frame.pack_forget()
                self.right_single_label.pack(fill=tk.BOTH, expand=True)
                
                method_name = self.method_var.get()
                func = METHODS[method_name]["func"]
                
                params = {k: self.param_vars[k].get() for k in METHODS[method_name]["params"]}
                if "binary_thresh" in params and params["binary_thresh"] == 0:
                    params["binary_thresh"] = None # Let Otsu handle it
                    
                if method_name == "Multi-scale Fusion":
                    params["canny_pairs"] = [(30, 90), (50, 150), (80, 200)] 
                
                res = func(self.grey_np, params)
                
                if res['success']:
                    out_rgb, txt = self._draw_result(self.color_np, res, method_name)
                    self.export_image_pil = Image.fromarray(out_rgb)
                    self.display_image(self.export_image_pil, self.right_single_label, max_size=(800, 800))
                    
                    self.status_var.set(f"Success! {txt} | Time: {res['diagnostics']['elapsed_ms']:.1f}ms")
                else:
                    self.export_image_pil = None
                    self.right_single_label.config(image="", text="Detection Failed:\n" + res['diagnostics']['message'])
                    messagebox.showerror("Failed", "Current method failed to detect a valid circle.\nTry adjusting the parameters.")
                    self.status_var.set(f"Failed: {res['diagnostics']['message']}")
                    
            else:
                self.right_single_label.pack_forget()
                self.right_grid_frame.pack(fill=tk.BOTH, expand=True)
                
                success_count = 0
                h, w, c = self.color_np.shape
                collage = np.zeros((h*2, w*2, c), dtype=np.uint8)
                
                for idx, (m_name, m_info) in enumerate(METHODS.items()):
                    func = m_info["func"]
                    m_params = {k: self.param_vars[k].get() for k in m_info["params"]}
                    if "binary_thresh" in m_params and m_params["binary_thresh"] == 0:
                        m_params["binary_thresh"] = None
                    if m_name == "Multi-scale Fusion":
                        m_params["canny_pairs"] = [(30, 90), (50, 150), (80, 200)]
                        
                    res = func(self.grey_np, m_params)
                    out_rgb, txt = self._draw_result(self.color_np, res, m_name)
                    
                    if res['success']: success_count += 1
                    
                    pil_img = Image.fromarray(out_rgb)
                    self.display_image(pil_img, self.grid_labels[idx], max_size=(400, 400))
                    
                    row, col = idx // 2, idx % 2
                    collage[row*h:(row+1)*h, col*w:(col+1)*w] = out_rgb
                    
                self.export_image_pil = Image.fromarray(collage)
                self.status_var.set(f"Compare Mode completed. {success_count}/4 algorithms found circles.")
                
            self.btn_export.config(state=tk.NORMAL if self.export_image_pil else tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Exception", f"An unexpected error occurred: {str(e)}")
        finally:
            self.root.config(cursor="")
            
    def export_results(self):
        if not self.export_image_pil:
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG file", "*.png")])
        if path:
            self.export_image_pil.save(path)
            messagebox.showinfo("Export Successful", f"Saved rendering to: {path}")
            self.status_var.set(f"Saved result block to {path}")
            
    def save_config(self):
        data = {k: v.get() for k, v in self.param_vars.items()}
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON file", "*.json")])
        if path:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            self.status_var.set(f"Config globally saved to {path}")
            
    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON file", "*.json")])
        if path:
            with open(path, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                if k in self.param_vars:
                    self.param_vars[k].set(v)
            self.status_var.set(f"Parameters locally updated from {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CircleApp(root)
    root.mainloop()
