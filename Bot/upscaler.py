import os
os.chdir("./Bot")
import PIL.Image


def upscale(image_path: str, output_path: str) -> str:
    os.system(
        f'./realesrgan/realesrgan-ncnn-vulkan.exe -i "{image_path}" -o "{output_path}"'
    )
    return output_path
