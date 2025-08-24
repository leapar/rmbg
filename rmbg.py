import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from engine import OnnxBaseModel

def open_img(img_path=None):
    """
    Convert 8bit/16bit RGB image or 8bit/16bit Gray image to 8bit RGB image
    """
    if img_path is not None and os.path.exists(img_path):
        # Load Image From Path Directly
        # NOTE: Potential issue - unable to handle the flipped image.
        # Temporary workaround: cv_image = cv2.imread(img_path)
        cv_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # To uint8
    if cv_image.dtype != np.uint8:
        cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)
        cv_image = np.array(cv_image, dtype=np.uint8)
    # To RGB
    if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
        cv_image = cv2.merge([cv_image, cv_image, cv_image])
    return cv_image

def get_image_files(folder_path):
    """
    获取文件夹中所有支持的图片文件

    Args:
        folder_path: 文件夹路径

    Returns:
        list: 图片文件路径列表
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []

    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return []

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            image_files.append(file_path)

    return sorted(image_files)

class RMBG():
    """A class for removing backgrounds from images using BRIA RMBG model."""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle":"Model: Rectangle",
        }
        default_output_mode = "rectangle"

    def __init__(self,model_abs_path) -> None:
        # Run the parent class's init method
      
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                    "Model" + f"Could not download or initializemodel.",
            )
        self.device = "gpu"
        self.model_path = model_abs_path
        self.net = OnnxBaseModel(model_abs_path,self.device)
        
        self.model_version = 1.4 #1.4  2.0
        try:
            idx = model.index("2.0")
            if idx > 0:
                self.model_version = 2.0
        except ValueError:
            pass
        
        if self.device == "cpu" and self.model_version == 2.0:
            print(
                "⚠️ RMBG model running on CPU will be very slow. Please consider using GPU acceleration for better performance."
            )

        # Set default input shape for different versions
        if self.model_version == 2.0:
            self.input_shape = (1024, 1024)
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            self.input_shape = self.net.get_input_shape()[-2:]
            self.mean = 0.5
            self.std = 1.0
        
        self.save_dir = "./temp1/"
        self.file_ext = ".rmbg.png"

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Preprocessed image.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=2)
        image = cv2.resize(
            image, self.input_shape, interpolation=cv2.INTER_LINEAR
        )
        image = image.astype(np.float32) / 255.0

        if self.model_version >= 2.0:
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        else:
            image = (image - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def forward(self, blob):
        return self.net.get_ort_inference(blob, extract=True, squeeze=True)

    def postprocess(
        self, result: np.ndarray, original_size: tuple
    ) -> np.ndarray:
        """
        Postprocess the model output.

        Args:
            result (np.ndarray): Model output.
            original_size (tuple): Original image size (height, width).

        Returns:
            np.ndarray: Postprocessed image as a numpy array.
        """
        h, w = original_size
        resize_shape = (w, h)

        result = cv2.resize(
            np.squeeze(result),
            resize_shape,
            interpolation=cv2.INTER_LINEAR,
        )
        max_val, min_val = np.max(result), np.min(result)
        result = (result - min_val) / (max_val - min_val)
        return (result * 255).astype(np.uint8)

    def predict_shapes(self, image_path=None):
        """
        Remove the background from an image and save the result.

        Args:
            image (np.ndarray): Input image as a numpy array.
            image_path (str): Path to the input image.
        """

        try:
            image = open_img(image_path)
        except Exception as e:  # noqa
            print("Could not inference model")
            print(e)
            return []

        blob = self.preprocess(image)
        output = self.forward(blob)
        result_image = self.postprocess(output, image.shape[:2])

        # Create the final image with transparent background
        pil_mask = Image.fromarray(result_image)
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGBA")
        pil_mask = pil_mask.convert("L")

        # Create a new image with an alpha channel
        output_image = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        output_image.paste(pil_image, (0, 0), pil_mask)

        # Save the result
        image_dir_path = os.path.dirname(image_path)
        save_path = os.path.join(image_dir_path, "..", self.save_dir)
        save_path = os.path.realpath(save_path)
        os.makedirs(save_path, exist_ok=True)
        image_file_name = os.path.basename(image_path)
        save_name = os.path.splitext(image_file_name)[0] + self.file_ext
        save_file = os.path.join(save_path, save_name)
        print(save_file)
        output_image.save(save_file)

        return []

    def unload(self):
        del self.net

if __name__ == '__main__':
    model = "./bria-rmbg-1.4.onnx"
    model = "./bria-rmbg-2.0.onnx"
    rmg = RMBG(model)

    # 获取所有图片文件
    image_files = get_image_files("./")
    if not image_files:
        print("未找到支持的图片文件")
        os._exit(-1)

    print(f"找到 {len(image_files)} 个图片文件")
    print("支持的格式: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp")
    print("=" * 50)

    for i, image_path in enumerate(image_files, 1):
        rmg.predict_shapes(image_path)


    print(
        f"✅successfully: {"ok"}"
    )
# type: rmbg
# name: rmbg_v20-r20250530
# provider: BRIA-AI
# display_name: RMBG v2.0
# version: 2.0
# model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v3.0.0/bria-rmbg-2.0.onnx
# 
# type: rmbg
# name: rmbg_v14-r20240908
# provider: BRIA-AI
# display_name: RMBG v1.4
# model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.4.3/bria-rmbg-1.4.onnx

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime-gpu
# cudnn https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
# 现在完成以后要把cudnn的路径放入path，否则报错which depends on "cudnn64_9.dll" which is missing