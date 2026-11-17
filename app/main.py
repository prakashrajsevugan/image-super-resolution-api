import io
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse

from app.model import load_model


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Image Super Resolution API",
    description="2x image super-resolution using PyTorch",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# Model
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/best_model.pth"
)


try:

    model, device = load_model(
        MODEL_PATH
    )

except Exception as e:

    print("MODEL LOADING FAILED")
    print(str(e))

    raise


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Image Super Resolution API",
        "model": "PyTorch",
        "scale": "2x",
        "device": str(device),
        "status": "running"
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


# ============================================================
# Upscale
# ============================================================

@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be detected."
        )


    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )


    try:

        # ====================================================
        # Read image
        # ====================================================

        contents = await file.read()


        if len(contents) == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )


        print("--------------------------------")
        print("Received:", file.filename)
        print("Content type:", file.content_type)
        print("File size:", len(contents), "bytes")


        # ====================================================
        # PIL image
        # ====================================================

        image = Image.open(
            io.BytesIO(contents)
        )


        print(
            "Original size:",
            image.size
        )


        # ====================================================
        # Convert to grayscale
        # ====================================================

        image = image.convert("L")


        # ====================================================
        # PIL -> NumPy
        # ====================================================

        image_np = np.asarray(
            image,
            dtype=np.float32
        )


        # ====================================================
        # Normalize
        #
        # 0 ... 255
        #      ↓
        # 0 ... 1
        # ====================================================

        image_np = image_np / 255.0


        # ====================================================
        # NumPy -> Tensor
        #
        # H x W
        #
        # ->
        #
        # 1 x 1 x H x W
        # ====================================================

        tensor = torch.from_numpy(
            image_np
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)


        print(
            "Input tensor:",
            tuple(tensor.shape)
        )


        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        with torch.inference_mode():

            # Network predicts the residual
            residual = model(
                tensor
            )


            # =================================================
            # Bicubic base image
            #
            # This provides the basic 2x image.
            # The neural network predicts the details
            # that should be added to this image.
            # =================================================

            base = F.interpolate(
                tensor,

                scale_factor=2,

                mode="bicubic",

                align_corners=False
            )


            # =================================================
            # Residual learning
            # =================================================

            output = base + residual


        print(
            "Residual range:",
            float(residual.min()),
            float(residual.max())
        )


        print(
            "Base range:",
            float(base.min()),
            float(base.max())
        )


        print(
            "Output range before clamp:",
            float(output.min()),
            float(output.max())
        )


        # ====================================================
        # Remove batch/channel dimensions
        #
        # 1 x 1 x H x W
        #
        # ->
        #
        # H x W
        # ====================================================

        output = output.squeeze(
            0
        ).squeeze(
            0
        )


        # ====================================================
        # CPU
        # ====================================================

        output = output.cpu().numpy()


        # ====================================================
        # Clamp
        #
        # Prevent values outside valid image range.
        # ====================================================

        output = np.clip(
            output,
            0.0,
            1.0
        )


        # ====================================================
        # 0 ... 1
        #
        # ->
        #
        # 0 ... 255
        # ====================================================

        output = (
            output * 255.0
        )


        output = np.round(
            output
        ).astype(
            np.uint8
        )


        # ====================================================
        # NumPy -> PIL
        # ====================================================

        output_image = Image.fromarray(
            output,
            mode="L"
        )


        print(
            "Output size:",
            output_image.size
        )


        # ====================================================
        # Save image to memory
        # ====================================================

        buffer = io.BytesIO()


        output_image.save(
            buffer,
            format="PNG"
        )


        buffer.seek(0)


        print(
            "Image generated successfully."
        )

        print("--------------------------------")


        # ====================================================
        # Return image
        # ====================================================

        return StreamingResponse(
            buffer,

            media_type="image/png",

            headers={
                "Content-Disposition":
                    'inline; filename="upscaled.png"'
            }
        )


    except HTTPException:

        raise


    except Exception as e:

        print("--------------------------------")
        print("INFERENCE ERROR")
        print(str(e))
        print("--------------------------------")


        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )