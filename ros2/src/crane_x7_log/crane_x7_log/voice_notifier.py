#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""Voice notification module using text-to-speech for data logger."""

import subprocess
import threading
from typing import Optional
from rclpy.node import Node


class VoiceNotifier:
    """Text-to-speech voice notifications for episode events."""

    def __init__(
        self,
        logger: Optional[Node] = None,
        enabled: bool = True,
        language: str = "en",
        speed: int = 150
    ):
        """
        Initialize voice notifier.

        Args:
            logger: ROS 2 logger for error reporting
            enabled: Enable/disable voice notifications
            language: Language code (e.g., 'en', 'ja')
            speed: Speech speed in words per minute
        """
        self.logger = logger
        self.enabled = enabled
        self.language = language
        self.speed = speed
        self._speaking = False

    def speak(self, text: str, async_mode: bool = True) -> None:
        """
        Speak the given text using text-to-speech.

        Args:
            text: Text to speak
            async_mode: If True, speak in background thread (non-blocking)
        """
        if not self.enabled:
            return

        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(text,))
            thread.daemon = True
            thread.start()
        else:
            self._speak_sync(text)

    def _speak_sync(self, text: str) -> None:
        """Synchronously speak text using espeak-ng."""
        if self._speaking:
            return

        self._speaking = True
        try:
            # Use espeak-ng for text-to-speech
            # -s: speed (words per minute)
            # -v: voice/language
            # --stdout: output to stdout (can be piped to aplay)
            subprocess.run(
                ["espeak-ng", "-s", str(self.speed), "-v", self.language, text],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0
            )
        except subprocess.TimeoutExpired:
            if self.logger:
                self.logger.get_logger().warn(f'Voice notification timeout: {text}')
        except FileNotFoundError:
            if self.logger:
                self.logger.get_logger().warn(
                    'espeak-ng not found. Voice notifications disabled.'
                )
            self.enabled = False
        except Exception as e:
            if self.logger:
                self.logger.get_logger().warn(f'Voice notification error: {e}')
        finally:
            self._speaking = False

    def notify_episode_start(self, episode_number: int) -> None:
        """Notify the start of a new episode."""
        self.speak(f"Episode {episode_number} started")

    def notify_episode_complete(self, episode_number: int) -> None:
        """Notify episode completion."""
        self.speak(f"Episode {episode_number} completed")

    def notify_time_remaining(self, seconds: int) -> None:
        """Notify remaining time."""
        if seconds >= 60:
            minutes = seconds // 60
            self.speak(f"{minutes} minute{'s' if minutes != 1 else ''} remaining")
        else:
            self.speak(f"{seconds} seconds remaining")

    def notify_resuming(self, episode_number: int, delay_seconds: float) -> None:
        """Notify that recording will resume after delay."""
        if delay_seconds >= 60:
            minutes = int(delay_seconds // 60)
            self.speak(
                f"Episode {episode_number - 1} saved. "
                f"Next episode starts in {minutes} minute{'s' if minutes != 1 else ''}"
            )
        else:
            self.speak(
                f"Episode {episode_number - 1} saved. "
                f"Next episode starts in {int(delay_seconds)} seconds"
            )

    def notify_custom(self, message: str) -> None:
        """Notify with custom message."""
        self.speak(message)
