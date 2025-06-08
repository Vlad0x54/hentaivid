# hentaivid/qr.py 
import qrcode
from PIL import Image
from typing import Optional
from loguru import logger
from pyzbar.pyzbar import decode as qr_decode

def create_qr_code(data: str, size: int) -> Optional[Image.Image]:
    """
    Creates a QR code image from the given data.
    """
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.error(f"Error creating QR code: {e}")
        return None 

def read_qr_code(image: Image.Image) -> Optional[str]:
    """
    Reads a QR code from a PIL image and returns the decoded data.
    """
    try:
        decoded_objects = qr_decode(image)
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')
        return None
    except Exception as e:
        logger.error(f"Error reading QR code: {e}")
        return None 