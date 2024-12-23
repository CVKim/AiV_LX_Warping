import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

class ImageNormalizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Normalizer")
        self.master.geometry("800x600")  # 초기 창 크기 설정 (필요에 따라 조정 가능)
        self.master.resizable(True, True)  # 창 크기 조정 가능하게 설정

        # Initialize variables
        self.original_image = None
        self.min_max_image = None
        self.mean_norm_image = None
        self.image_path = ""
        
        # Create UI components
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        btn_frame = tk.Frame(self.master)
        btn_frame.pack(pady=10)

        self.load_btn = tk.Button(btn_frame, text="이미지 불러오기", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=10)

        self.save_btn = tk.Button(btn_frame, text="정규화된 이미지 저장", command=self.save_images, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=10)

        # Frame for images with scrollbars
        img_frame = tk.Frame(self.master)
        img_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas and Scrollbars for Original Image
        orig_canvas_frame = tk.Frame(img_frame)
        orig_canvas_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        orig_label = tk.Label(orig_canvas_frame, text="원본 이미지")
        orig_label.pack()

        self.orig_canvas = tk.Canvas(orig_canvas_frame, bg='grey')
        self.orig_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        orig_scrollbar_y = tk.Scrollbar(orig_canvas_frame, orient=tk.VERTICAL, command=self.orig_canvas.yview)
        orig_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.orig_canvas.configure(yscrollcommand=orig_scrollbar_y.set)

        orig_scrollbar_x = tk.Scrollbar(orig_canvas_frame, orient=tk.HORIZONTAL, command=self.orig_canvas.xview)
        orig_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.orig_canvas.configure(xscrollcommand=orig_scrollbar_x.set)

        # Canvas and Scrollbars for Min-Max Normalized Image
        min_max_canvas_frame = tk.Frame(img_frame)
        min_max_canvas_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        min_max_label = tk.Label(min_max_canvas_frame, text="Min-Max 정규화 이미지")
        min_max_label.pack()

        self.min_max_canvas = tk.Canvas(min_max_canvas_frame, bg='grey')
        self.min_max_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        min_max_scrollbar_y = tk.Scrollbar(min_max_canvas_frame, orient=tk.VERTICAL, command=self.min_max_canvas.yview)
        min_max_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.min_max_canvas.configure(yscrollcommand=min_max_scrollbar_y.set)

        min_max_scrollbar_x = tk.Scrollbar(min_max_canvas_frame, orient=tk.HORIZONTAL, command=self.min_max_canvas.xview)
        min_max_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.min_max_canvas.configure(xscrollcommand=min_max_scrollbar_x.set)

        # Canvas and Scrollbars for Mean Normalized Image
        mean_norm_canvas_frame = tk.Frame(img_frame)
        mean_norm_canvas_frame.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')

        mean_norm_label = tk.Label(mean_norm_canvas_frame, text="Mean 정규화 이미지")
        mean_norm_label.pack()

        self.mean_norm_canvas = tk.Canvas(mean_norm_canvas_frame, bg='grey')
        self.mean_norm_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        mean_norm_scrollbar_y = tk.Scrollbar(mean_norm_canvas_frame, orient=tk.VERTICAL, command=self.mean_norm_canvas.yview)
        mean_norm_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.mean_norm_canvas.configure(yscrollcommand=mean_norm_scrollbar_y.set)

        mean_norm_scrollbar_x = tk.Scrollbar(mean_norm_canvas_frame, orient=tk.HORIZONTAL, command=self.mean_norm_canvas.xview)
        mean_norm_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.mean_norm_canvas.configure(xscrollcommand=mean_norm_scrollbar_x.set)

        # Configure grid weights for proper resizing
        img_frame.grid_rowconfigure(0, weight=1)
        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_columnconfigure(1, weight=1)
        img_frame.grid_columnconfigure(2, weight=1)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            # Load original image
            self.original_image = Image.open(file_path).convert("RGB")
            self.image_path = file_path
            self.display_image(self.original_image, self.orig_canvas)

            # Convert to NumPy array
            img_array = np.array(self.original_image).astype(np.float32)

            # Min-Max Normalization
            min_val = img_array.min(axis=(0,1), keepdims=True)
            max_val = img_array.max(axis=(0,1), keepdims=True)
            min_max_norm = (img_array - min_val) / (max_val - min_val + 1e-8) * 255
            min_max_norm = min_max_norm.clip(0, 255).astype(np.uint8)
            self.min_max_image = Image.fromarray(min_max_norm)
            self.display_image(self.min_max_image, self.min_max_canvas)

            # Mean (Z-Score) Normalization
            mean = img_array.mean(axis=(0,1), keepdims=True)
            std = img_array.std(axis=(0,1), keepdims=True)
            mean_norm = (img_array - mean) / (std + 1e-8)  # Z-Score
            # Scale to 0-255 for display
            mean_norm = mean_norm - mean_norm.min()
            mean_norm = mean_norm / (mean_norm.max() + 1e-8) * 255
            mean_norm = mean_norm.clip(0, 255).astype(np.uint8)
            self.mean_norm_image = Image.fromarray(mean_norm)
            self.display_image(self.mean_norm_image, self.mean_norm_canvas)

            # Enable save button
            self.save_btn.config(state=tk.NORMAL)

        except Exception as e:
            import traceback
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            messagebox.showerror("에러", f"이미지 로드 실패: {e}\n{traceback_str}")

    def display_image(self, pil_image, canvas):
        # Convert PIL Image to ImageTk
        tk_image = ImageTk.PhotoImage(pil_image)

        # Clear previous image
        canvas.delete("all")

        # Add image to canvas
        canvas.create_image(0, 0, anchor='nw', image=tk_image)
        canvas.image = tk_image  # Keep a reference to avoid garbage collection

        # Configure scroll region
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

    def save_images(self):
        if not self.image_path:
            messagebox.showwarning("경고", "저장할 이미지가 없습니다.")
            return

        try:
            # Get directory of the original image
            dir_path = os.path.dirname(self.image_path)
            result_dir = os.path.join(dir_path, "result_norm")
            os.makedirs(result_dir, exist_ok=True)

            # Save min-max normalized image
            min_max_path = os.path.join(result_dir, "min_max_normalized.jpg")
            self.min_max_image.save(min_max_path)

            # Save mean normalized image
            mean_norm_path = os.path.join(result_dir, "mean_normalized.jpg")
            self.mean_norm_image.save(mean_norm_path)

            messagebox.showinfo("성공", f"이미지가 성공적으로 저장되었습니다:\n{result_dir}")

        except Exception as e:
            import traceback
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            messagebox.showerror("에러", f"이미지 저장 실패: {e}\n{traceback_str}")

def main():
    root = tk.Tk()
    app = ImageNormalizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
