import hashlib

from app.config import settings


def upload_image(raw: bytes, filename: str, content_type: str) -> str | None:
    if not settings.store_images or not settings.s3_bucket:
        return None
    import boto3
    key = f"predictions/{hashlib.sha256(raw).hexdigest()}-{filename}"
    client = boto3.client(
        "s3", region_name=settings.s3_region, endpoint_url=settings.s3_endpoint_url
    )
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=raw,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )
    return f"s3://{settings.s3_bucket}/{key}"
