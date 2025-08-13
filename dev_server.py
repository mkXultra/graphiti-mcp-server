#!/usr/bin/env python
"""開発用ホットリロードサーバー"""

import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RestartHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_server()
    
    def start_server(self):
        """サーバーを起動"""
        if self.process:
            print("🔄 サーバーを再起動中...")
            self.process.terminate()
            self.process.wait()
        
        print("🚀 サーバーを起動中...")
        self.process = subprocess.Popen(
            ["uv", "run", "graphiti_mcp_server.py", "--transport", "sse"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    
    def on_modified(self, event):
        """ファイル変更時の処理"""
        if event.src_path.endswith('.py'):
            print(f"📝 変更を検知: {event.src_path}")
            self.start_server()

if __name__ == "__main__":
    handler = RestartHandler()
    observer = Observer()
    observer.schedule(handler, path='.', recursive=True)
    observer.start()
    
    print("👀 ファイル変更を監視中... (Ctrl+Cで終了)")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
        if handler.process:
            handler.process.terminate()
    observer.join()