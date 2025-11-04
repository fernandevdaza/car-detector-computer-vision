from app.core.agent import agent
from fastapi import File
import base64
from langchain.messages import HumanMessage, TextContentBlock, ImageContentBlock


async def infer_with_image_controller(file: File):
    data = await file.read()
    b64_bytes = base64.b64encode(data)
    b64_str = b64_bytes.decode("utf-8")
    content_blocks = [
        TextContentBlock(type="text", text="Analiza esta imagen y dime qué auto es, marca, modelo, año aproximado, es un auto comprado en Bolivia"),
        ImageContentBlock(type="image", base64=b64_str, mime_type="image/jpeg")
    ]

    return await agent.ainvoke({"messages": [HumanMessage(content=content_blocks)]})
    # return content_blocks