"""HeyGen MCP server module for providing MCP tools for the HeyGen API."""

import argparse
import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.models import Resource
from mcp.server.models import Resource

from heygen_mcp.api_client import (
    Character,
    Dimension,
    HeyGenApiClient,
    MCPAvatarGroupResponse,
    MCPAvatarsInGroupResponse,
    MCPGetCreditsResponse,
    MCPVideoGenerateResponse,
    MCPVideoStatusResponse,
    MCPVoicesResponse,
    VideoGenerateRequest,
    VideoInput,
    Voice,
)

# Load environment variables
load_dotenv()

# Create MCP server instance
mcp = FastMCP("HeyGen MCP")
api_client = None


# Function to get or create API client
async def get_api_client() -> HeyGenApiClient:
    """Get the API client, creating it if necessary."""
    global api_client

    # If we already have a client, return it
    if api_client is not None:
        return api_client

    # Otherwise, get the API key and create a new client
    api_key = os.getenv("HEYGEN_API_KEY")

    if not api_key:
        raise ValueError("HEYGEN_API_KEY environment variable not set.")

    # Create and store the client
    api_client = HeyGenApiClient(api_key)
    return api_client


########################
# MCP Tool Definitions #
########################


@mcp.tool(
    name="account_credits",
    description="Retrieves the remaining credits in your HeyGen account.",
)
async def account_credits() -> MCPGetCreditsResponse:
    """Get the remaining quota for the user via HeyGen API."""
    try:
        client = await get_api_client()
        return await client.get_remaining_credits()
    except Exception as e:
        return MCPGetCreditsResponse(error=str(e))


@mcp.tool(
    name="list_available_voices",
    description=(
        "Retrieves a list of available voices from the HeyGen API. Results truncated "
        "to first 100 voices. Private voices generally will returned 1st."
    ),
)
async def list_available_voices() -> MCPVoicesResponse:
    """Get the list of available voices via HeyGen API."""
    try:
        client = await get_api_client()
        return await client.get_voices()
    except Exception as e:
        return MCPVoicesResponse(error=str(e))


@mcp.tool(
    name="list_avatar_groups",
    description=(
        "Retrieves a list of HeyGen avatar groups. By default, only private avatar "
        "groups are returned, unless include_public is set to true. Avatar groups "
        "are collections of avatars, avatar group ids cannot be used to generate "
        "videos."
    ),
)
async def list_avatar_groups(include_public: bool = False) -> MCPAvatarGroupResponse:
    """List avatar groups via HeyGen API v2/avatar_group.list endpoint."""
    try:
        client = await get_api_client()
        return await client.list_avatar_groups(include_public)
    except Exception as e:
        return MCPAvatarGroupResponse(error=str(e))


@mcp.tool(
    name="list_avatars_in_group",
    description="Retrieves a list of avatars in a specific HeyGen avatar group.",
)
async def list_avatars_in_group(group_id: str) -> MCPAvatarsInGroupResponse:
    """List avatars in a specific HeyGen avatar group via HeyGen API."""
    try:
        client = await get_api_client()
        return await client.get_avatars_in_group(group_id)
    except Exception as e:
        return MCPAvatarsInGroupResponse(error=str(e))


@mcp.tool(
    name="create_video",
    description="Step 1: Create a new avatar video with HeyGen API. Input text is required, avatar and voice are optional.",
)
async def create_video(
    input_text: str, avatar_id: Optional[str] = None, voice_id: Optional[str] = None, title: str = ""
) -> MCPVideoGenerateResponse:
    """Generate a new avatar video using the HeyGen API."""
    try:
        # Create the request object with default values
        request = VideoGenerateRequest(
            title=title,
            video_inputs=[
                VideoInput(
                    character=Character(avatar_id=avatar_id),
                    voice=Voice(input_text=input_text, voice_id=voice_id),
                )
            ],
            dimension=Dimension(width=1280, height=720),
        )

        client = await get_api_client()
        return await client.generate_avatar_video(request)
    except Exception as e:
        return MCPVideoGenerateResponse(error=str(e))


@mcp.tool(
    name="check_video_status",
    description=(
        "Step 2: Check the status of a video being generated. Video processing "
        "may take several minutes to hours depending on length and queue time."
    ),
)
async def check_video_status(video_id: str) -> MCPVideoStatusResponse:
    """Retrieve the status of a video generated via the HeyGen API."""
    try:
        client = await get_api_client()
        return await client.get_video_status(video_id)
    except Exception as e:
        return MCPVideoStatusResponse(error=str(e))


@mcp.tool(
    name="wait_for_video",
    description=(
        "Step 3: Wait for a video to complete processing. This tool will poll the "
        "video status API until the video is complete or fails."
    ),
)
async def wait_for_video(
    video_id: str, max_attempts: int = 60, delay_seconds: int = 10
) -> MCPVideoStatusResponse:
    """Wait for a video to complete processing, polling the status endpoint."""
    try:
        client = await get_api_client()
        return await client.wait_for_video_completion(video_id, max_attempts, delay_seconds)
    except Exception as e:
        return MCPVideoStatusResponse(error=str(e))


@mcp.tool(
    name="download_video",
    description=(
        "Step 4: Download a completed video to a local file and return the file "
        "path and a resource link."
    ),
)
async def download_video(video_id: str, download_path: Optional[str] = None) -> MCPVideoGenerateResponse:
    """Download a completed video to a local file."""
    try:
        client = await get_api_client()
        
        # First get the current status to ensure we have the video URL
        status = await client.get_video_status(video_id)
        
        if not status.video_url:
            return MCPVideoGenerateResponse(
                error="Video URL not available. Video may still be processing or failed."
            )

        # Generate a download path if not provided
        if not download_path:
            # Create temp dir if it doesn't exist
            temp_dir = Path(tempfile.gettempdir()) / "heygen_videos"
            temp_dir.mkdir(exist_ok=True)
            
            # Generate a filename using timestamp and video_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_path = str(temp_dir / f"{timestamp}_{video_id}.mp4")
        
        # Download the video
        file_path = await client.download_video(status.video_url, download_path)
        
        # Create an MCP resource with the local file path
        resource = Resource(file_path)
        
        # Return the download information
        return MCPVideoGenerateResponse(
            video_id=video_id,
            status="downloaded",
            video_url=status.video_url,
            download_path=file_path,
            resource=resource,
        )
    except Exception as e:
        return MCPVideoGenerateResponse(error=str(e))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="HeyGen MCP Server")
    parser.add_argument(
        "--api-key",
        help=(
            "HeyGen API key. Alternatively, set HEYGEN_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to."
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    return parser.parse_args()


def main():
    """Run the MCP server."""
    args = parse_args()

    # Check if API key is provided or in environment
    if args.api_key:
        os.environ["HEYGEN_API_KEY"] = args.api_key

    # Verify API key is set
    if not os.getenv("HEYGEN_API_KEY"):
        print("ERROR: HeyGen API key not provided.")
        print(
            "Please set it using --api-key or the HEYGEN_API_KEY environment variable."
        )
        sys.exit(1)

    mcp.run()


if __name__ == "__main__":
    main()
