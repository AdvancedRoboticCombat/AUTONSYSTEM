import { io } from "socket.io-client";

// Connect to the Python backend on port 3001
export const socket = io("http://localhost:3001", {
    transports: ["websocket"], // Force WebSocket for better performance
    autoConnect: true,
});