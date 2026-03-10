from __future__ import annotations

from pathlib import Path
import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import cv2
from inference_sdk import InferenceHTTPClient


DEFAULT_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
DEFAULT_API_KEY = os.getenv("ROBOFLOW_API_KEY", "0tD81VP5ij7wTYq6W7yA")
DEFAULT_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "1-v8mk1")
DEFAULT_WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW", "custom-workflow-2")


def extract_predictions(payload):
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return []

    predictions_block = payload.get("predictions", payload)
    if not isinstance(predictions_block, dict):
        return []

    predictions = predictions_block.get("predictions", [])
    return predictions if isinstance(predictions, list) else []


def draw_predictions(image_path: Path, output_path: Path, predictions):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    img_h, img_w = image.shape[:2]

    for prediction in predictions:
        x = prediction.get("x")
        y = prediction.get("y")
        w = prediction.get("width")
        h = prediction.get("height")
        if None in (x, y, w, h):
            continue

        x1 = max(0, int(round(x - w / 2)))
        y1 = max(0, int(round(y - h / 2)))
        x2 = min(img_w - 1, int(round(x + w / 2)))
        y2 = min(img_h - 1, int(round(y + h / 2)))

        label = str(prediction.get("class", "unknown"))
        confidence = prediction.get("confidence")
        if isinstance(confidence, (float, int)):
            text = f"{label} {confidence:.2f}"
        else:
            text = label

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_y1 = max(0, y1 - text_h - baseline - 6)
        text_y2 = max(text_h + baseline + 6, y1)
        text_x2 = min(img_w - 1, x1 + text_w + 8)

        cv2.rectangle(image, (x1, text_y1), (text_x2, text_y2), (0, 220, 0), -1)
        cv2.putText(
            image,
            text,
            (x1 + 4, text_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


class InferenceGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TP2 Inference GUI")
        self.root.geometry("900x620")

        self.selected_images: list[Path] = []
        self.output_dir = Path.cwd()

        self._build_ui()

    def _build_ui(self):
        cfg_frame = tk.LabelFrame(self.root, text="Configuracion Roboflow")
        cfg_frame.pack(fill="x", padx=10, pady=8)

        self.api_url_var = tk.StringVar(value=DEFAULT_API_URL)
        self.api_key_var = tk.StringVar(value=DEFAULT_API_KEY)
        self.workspace_var = tk.StringVar(value=DEFAULT_WORKSPACE)
        self.workflow_var = tk.StringVar(value=DEFAULT_WORKFLOW)

        self._add_labeled_entry(cfg_frame, "API URL", self.api_url_var, 0)
        self._add_labeled_entry(cfg_frame, "API Key", self.api_key_var, 1, show="*")
        self._add_labeled_entry(cfg_frame, "Workspace", self.workspace_var, 2)
        self._add_labeled_entry(cfg_frame, "Workflow", self.workflow_var, 3)

        sel_frame = tk.LabelFrame(self.root, text="Imagenes")
        sel_frame.pack(fill="x", padx=10, pady=8)

        btn_row = tk.Frame(sel_frame)
        btn_row.pack(fill="x", padx=6, pady=6)

        tk.Button(btn_row, text="Seleccionar imagen(es)", command=self.select_images).pack(
            side="left", padx=4
        )
        tk.Button(btn_row, text="Elegir carpeta salida", command=self.select_output_dir).pack(
            side="left", padx=4
        )
        tk.Button(btn_row, text="Ejecutar inferencia", command=self.run_inference).pack(
            side="left", padx=4
        )

        self.output_dir_label = tk.Label(
            sel_frame, text=f"Salida: {self.output_dir}", anchor="w"
        )
        self.output_dir_label.pack(fill="x", padx=10, pady=(0, 8))

        list_frame = tk.LabelFrame(self.root, text="Archivos seleccionados")
        list_frame.pack(fill="both", expand=True, padx=10, pady=8)

        self.listbox = tk.Listbox(list_frame, height=10)
        self.listbox.pack(fill="both", expand=True, padx=8, pady=8)

        log_frame = tk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)

        self.log = scrolledtext.ScrolledText(log_frame, height=12, state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def _add_labeled_entry(self, parent, label, var, row, show=None):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        entry = tk.Entry(parent, textvariable=var, width=90, show=show)
        entry.grid(row=row, column=1, sticky="we", padx=8, pady=4)
        parent.grid_columnconfigure(1, weight=1)

    def _append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")
        self.root.update_idletasks()

    def select_images(self):
        filetypes = [
            ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Todos", "*.*"),
        ]
        files = filedialog.askopenfilenames(title="Selecciona una o varias imagenes", filetypes=filetypes)
        if not files:
            return

        self.selected_images = [Path(f) for f in files]
        self.listbox.delete(0, tk.END)
        for path in self.selected_images:
            self.listbox.insert(tk.END, str(path))

        self._append_log(f"Seleccionadas {len(self.selected_images)} imagen(es).")

    def select_output_dir(self):
        selected = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if not selected:
            return
        self.output_dir = Path(selected)
        self.output_dir_label.configure(text=f"Salida: {self.output_dir}")
        self._append_log(f"Carpeta de salida: {self.output_dir}")

    def _build_client(self):
        return InferenceHTTPClient(
            api_url=self.api_url_var.get().strip(),
            api_key=self.api_key_var.get().strip(),
        )

    def run_inference(self):
        if not self.selected_images:
            messagebox.showwarning("Sin imagenes", "Selecciona al menos una imagen.")
            return

        api_url = self.api_url_var.get().strip()
        api_key = self.api_key_var.get().strip()
        workspace = self.workspace_var.get().strip()
        workflow = self.workflow_var.get().strip()

        if not all([api_url, api_key, workspace, workflow]):
            messagebox.showerror("Configuracion incompleta", "Revisa API URL, API Key, Workspace y Workflow.")
            return

        self._append_log("Iniciando inferencia...")

        try:
            client = self._build_client()
        except Exception as exc:
            messagebox.showerror("Error cliente", str(exc))
            return

        processed = 0
        errors = 0

        for image_path in self.selected_images:
            try:
                self._append_log(f"Procesando: {image_path}")
                result = client.run_workflow(
                    workspace_name=workspace,
                    workflow_id=workflow,
                    images={"image": str(image_path)},
                    use_cache=True,
                )
                predictions = extract_predictions(result)

                output_name = f"{image_path.stem}_pred{image_path.suffix}"
                output_path = self.output_dir / output_name
                draw_predictions(image_path, output_path, predictions)

                self._append_log(
                    f"OK | detecciones={len(predictions)} | salida={output_path}"
                )
                processed += 1
            except Exception:
                errors += 1
                self._append_log(f"ERROR en {image_path}")
                self._append_log(traceback.format_exc().strip())

        summary = f"Finalizado. Correctas: {processed}, Errores: {errors}"
        self._append_log(summary)
        messagebox.showinfo("TP2 Inference GUI", summary)


def main():
    root = tk.Tk()
    InferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
