"""HeyGen API client module for interacting with the HeyGen API."""

import importlib.metadata
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, HttpUrl

#######################
# HeyGen API Models #
#######################


# Common base response model
class BaseHeyGenResponse(BaseModel):
    error: Optional[str] = None


# Voice information models
class VoiceInfo(BaseModel):
    voice_id: str
    language: str
    gender: str
    name: str
    preview_audio: HttpUrl
    support_pause: bool
    emotion_support: bool
    support_interactive_avatar: bool


class VoicesData(BaseModel):
    voices: List[VoiceInfo]


class VoicesResponse(BaseHeyGenResponse):
    data: Optional[VoicesData] = None


# User quota models
class QuotaDetails(BaseModel):
    api: int
    streaming_avatar: int
    streaming_avatar_instance_quota: int
    seat: int


class RemainingQuota(BaseModel):
    remaining_quota: int
    details: QuotaDetails


class RemainingQuotaResponse(BaseHeyGenResponse):
    data: Optional[RemainingQuota] = None


# Avatar group models
class AvatarGroup(BaseModel):
    id: str
    name: str
    created_at: int
    num_looks: int
    preview_image: HttpUrl
    group_type: str
    train_status: Optional[str] = None


class AvatarGroupListData(BaseModel):
    total_count: int
    avatar_group_list: List[AvatarGroup]


class AvatarGroupListResponse(BaseHeyGenResponse):
    data: Optional[AvatarGroupListData] = None


# Avatar models
class Avatar(BaseModel):
    avatar_id: str
    avatar_name: str
    gender: str
    preview_image_url: HttpUrl
    preview_video_url: HttpUrl
    premium: bool
    type: Optional[str] = None
    tags: Optional[List[str]] = None
    default_voice_id: Optional[str] = None


class AvatarsInGroupData(BaseModel):
    avatar_list: List[Avatar]


class AvatarsInGroupResponse(BaseHeyGenResponse):
    data: Optional[AvatarsInGroupData] = None


# Video generation models
class Character(BaseModel):
    type: str = "avatar"
    avatar_id: str
    avatar_style: str = "normal"
    scale: float = 1.0


class Voice(BaseModel):
    type: str = "text"
    input_text: str
    voice_id: str


class VideoInput(BaseModel):
    character: Character
    voice: Voice


class Dimension(BaseModel):
    width: int = 1280
    height: int = 720


class VideoGenerateRequest(BaseModel):
    title: str = ""
    video_inputs: List[VideoInput]
    test: bool = False
    callback_id: Optional[str] = None
    dimension: Dimension = Field(default_factory=lambda: Dimension())
    aspect_ratio: Optional[str] = None
    caption: bool = False


class VideoGenerateResponse(BaseHeyGenResponse):
    data: Optional[Dict[str, Any]] = None


# Video status models
class VideoStatusError(BaseModel):
    code: Optional[int] = None
    detail: Optional[str] = None
    message: Optional[str] = None


class VideoStatusData(BaseModel):
    callback_id: Optional[str] = None
    caption_url: Optional[str] = None
    created_at: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[VideoStatusError] = None
    gif_url: Optional[str] = None
    id: str
    status: str  # Values: "waiting", "pending", "processing", "completed", "failed"
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None
    video_url_caption: Optional[str] = None


class VideoStatusResponse(BaseModel):
    code: int
    data: VideoStatusData
    message: str


########################
# MCP Response Models #
########################


class MCPGetCreditsResponse(BaseHeyGenResponse):
    remaining_credits: Optional[int] = None


class MCPVoicesResponse(BaseHeyGenResponse):
    voices: Optional[List[VoiceInfo]] = None


class MCPAvatarGroupResponse(BaseHeyGenResponse):
    avatar_groups: Optional[List[AvatarGroup]] = None
    total_count: Optional[int] = None


class MCPAvatarsInGroupResponse(BaseHeyGenResponse):
    avatars: Optional[List[Avatar]] = None


class MCPVideoGenerateResponse(BaseHeyGenResponse):
    video_id: Optional[str] = None
    task_id: Optional[str] = None
    video_url: Optional[str] = None
    status: Optional[str] = None


class MCPVideoStatusResponse(BaseHeyGenResponse):
    video_id: Optional[str] = None
    status: Optional[str] = None
    duration: Optional[float] = None
    video_url: Optional[str] = None
    gif_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: Optional[int] = None
    error_details: Optional[Dict[str, Any]] = None


# HeyGen API Client Class
class HeyGenApiClient:
    """Client for interacting with the HeyGen API."""

    def __init__(self, api_key: str):
        """Initialize the API client with the API key."""
        self.api_key = api_key

        # Set version for user agent
        try:
            self.version = importlib.metadata.version("heygen-mcp")
        except importlib.metadata.PackageNotFoundError:
            self.version = "unknown"

        self.user_agent = f"heygen-mcp/{self.version}"
        self.base_url = "https://api.heygen.com/v2"
        self._client = httpx.AsyncClient()

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _get_headers(self) -> Dict[str, str]:
        """Return the headers needed for API requests."""
        return {
            "Accept": "application/json",
            "X-Api-Key": self.api_key,
            "User-Agent": self.user_agent,
        }

    async def _make_request(
        self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a request to the specified API endpoint.

        Args:
            endpoint: The API endpoint to call (without the base URL)
            method: HTTP method to use (GET or POST)
            data: JSON payload for POST requests

        Returns:
            The JSON response from the API

        Raises:
            httpx.RequestError: If there's a network-related error
            httpx.HTTPStatusError: If the API returns an error status code
            Exception: For any other unexpected errors
        """
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()

        if method.upper() == "GET":
            response = await self._client.get(url, headers=headers)
        elif method.upper() == "POST":
            headers["Content-Type"] = "application/json"
            response = await self._client.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()  # Raises if status code is 4xx or 5xx
        return response.json()

    async def _handle_api_request(
        self,
        api_call,
        response_model_class,
        mcp_response_class,
        error_msg: str,
        **kwargs,
    ):
        """Generic handler for API requests to reduce code duplication.

        Args:
            api_call: Async function to call the API
            response_model_class: Pydantic model class for validating the API response
            mcp_response_class: Pydantic model class for the MCP response
            error_msg: Error message to return if the validation fails
            **kwargs: Additional arguments for the response transformation

        Returns:
            An MCP response object
        """
        try:
            # Make the request to the API
            result = await api_call()

            # Validate the response
            validated_response = response_model_class.model_validate(result)

            # Return the appropriate response based on the validation result
            if hasattr(validated_response, "data") and validated_response.data:
                return self._transform_to_mcp_response(
                    validated_response.data, mcp_response_class, **kwargs
                )
            elif validated_response.error:
                return mcp_response_class(error=validated_response.error)
            else:
                return mcp_response_class(error=error_msg)

        except httpx.RequestError as exc:
            return mcp_response_class(error=f"HTTP Request failed: {exc}")
        except httpx.HTTPStatusError as exc:
            return mcp_response_class(
                error=f"HTTP Error: {exc.response.status_code} - {exc.response.text}"
            )
        except Exception as e:
            return mcp_response_class(error=f"An unexpected error occurred: {e}")

    def _transform_to_mcp_response(self, data, mcp_response_class, **kwargs):
        """Transform API response data to MCP response format.

        Args:
            data: The API response data
            mcp_response_class: The MCP response class to instantiate
            **kwargs: Additional parameters for the response

        Returns:
            An instance of the MCP response class
        """
        if "transform_func" in kwargs:
            # Use the provided transform function
            transform_func = kwargs.pop("transform_func")
            return transform_func(data, mcp_response_class)

        # Apply lambda functions to data if provided or use direct values
        processed_kwargs = {}
        for key, value in kwargs.items():
            if callable(value):
                processed_kwargs[key] = value(data)
            else:
                processed_kwargs[key] = value

        return mcp_response_class(**processed_kwargs)

    async def get_remaining_credits(self) -> MCPGetCreditsResponse:
        """Get the remaining credits from the API."""

        async def api_call():
            return await self._make_request("user/remaining_quota")

        def transform_data(data, mcp_class):
            return mcp_class(remaining_credits=int(data.remaining_quota / 60))

        return await self._handle_api_request(
            api_call=api_call,
            response_model_class=RemainingQuotaResponse,
            mcp_response_class=MCPGetCreditsResponse,
            error_msg="No quota information found.",
            transform_func=transform_data,
        )

    async def get_voices(self) -> MCPVoicesResponse:
        """Get the list of available voices from the API."""

        async def api_call():
            return await self._make_request("voices")

        def transform_data(data, mcp_class):
            # Truncate to the first 100 voices
            return mcp_class(voices=data.voices[:100] if data.voices else None)

        return await self._handle_api_request(
            api_call=api_call,
            response_model_class=VoicesResponse,
            mcp_response_class=MCPVoicesResponse,
            error_msg="No voices found.",
            transform_func=transform_data,
        )

    async def list_avatar_groups(
        self, include_public: bool = False
    ) -> MCPAvatarGroupResponse:
        """Get the list of avatar groups from the API."""

        async def api_call():
            public_param = "true" if include_public else "false"
            endpoint = f"avatar_group.list?include_public={public_param}"
            return await self._make_request(endpoint)

        def transform_data(data, mcp_class):
            return mcp_class(
                avatar_groups=data.avatar_group_list, total_count=data.total_count
            )

        return await self._handle_api_request(
            api_call=api_call,
            response_model_class=AvatarGroupListResponse,
            mcp_response_class=MCPAvatarGroupResponse,
            error_msg="No avatar groups found.",
            transform_func=transform_data,
        )

    async def get_avatars_in_group(self, group_id: str) -> MCPAvatarsInGroupResponse:
        """Get the list of avatars in a specific avatar group."""

        async def api_call():
            endpoint = f"avatar_group/{group_id}/avatars"
            return await self._make_request(endpoint)

        def transform_data(data, mcp_class):
            return mcp_class(avatars=data.avatar_list)

        return await self._handle_api_request(
            api_call=api_call,
            response_model_class=AvatarsInGroupResponse,
            mcp_response_class=MCPAvatarsInGroupResponse,
            error_msg="No avatars found in the group.",
            transform_func=transform_data,
        )

    async def generate_avatar_video(
        self, video_request: VideoGenerateRequest
    ) -> MCPVideoGenerateResponse:
        """Generate an avatar video using the HeyGen API."""

        async def api_call():
            return await self._make_request(
                "video/generate", method="POST", data=video_request.model_dump()
            )

        return await self._handle_api_request(
            api_call=api_call,
            response_model_class=VideoGenerateResponse,
            mcp_response_class=MCPVideoGenerateResponse,
            error_msg="No video generation data returned.",
            video_id=lambda d: d.get("video_id"),
            task_id=lambda d: d.get("task_id"),
            video_url=lambda d: d.get("video_url"),
            status=lambda d: d.get("status"),
        )

    async def generate_avatar_video_with_multiple_scenes(
        self, video_request: List[VideoGenerateRequest]
    ) -> MCPVideoGenerateResponse:
        """Generate an avatar video using the HeyGen API with multiple scenes"""

        async def api_call():
            return await self._make_request(
                "video/generate", method="POST", data=video_request.model_dump()
            )

        return await self._handle_api_request(
            api_call=api_call,
            response_model_class=VideoGenerateResponse,
            mcp_response_class=MCPVideoGenerateResponse,
            error_msg="No video generation data returned.",
            video_id=lambda d: d.get("video_id"),
            task_id=lambda d: d.get("task_id"),
            video_url=lambda d: d.get("video_url"),
            status=lambda d: d.get("status"),
        )

    async def get_video_status(self, video_id: str) -> MCPVideoStatusResponse:
        """Get the status of a generated video from the API."""

        async def api_call():
            # The endpoint is v1, not v2
            endpoint = f"../v1/video_status.get?video_id={video_id}"
            return await self._make_request(endpoint)

        try:
            # Make the request to the API
            result = await api_call()

            # Validate the response
            validated_response = VideoStatusResponse.model_validate(result)

            # Extract data
            data = validated_response.data

            # Process error details if present
            error_details = None
            if data.error:
                error_details = {
                    "code": data.error.code,
                    "message": data.error.message,
                    "detail": data.error.detail,
                }

            # Return MCP response
            return MCPVideoStatusResponse(
                video_id=data.id,
                status=data.status,
                duration=data.duration,
                video_url=data.video_url,
                gif_url=data.gif_url,
                thumbnail_url=data.thumbnail_url,
                created_at=data.created_at,
                error_details=error_details,
            )
        except httpx.RequestError as exc:
            return MCPVideoStatusResponse(error=f"HTTP Request failed: {exc}")
        except httpx.HTTPStatusError as exc:
            return MCPVideoStatusResponse(
                error=f"HTTP Error: {exc.response.status_code} - {exc.response.text}"
            )
        except Exception as e:
            return MCPVideoStatusResponse(error=f"An unexpected error occurred: {e}")
