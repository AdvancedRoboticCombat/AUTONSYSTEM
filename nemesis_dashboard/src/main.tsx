import { createRoot } from "react-dom/client";
import App from "./App.tsx";
// Fonts ship with the app: the dashboard PC is usually offline on the camera network
import "@fontsource/orbitron/500.css";
import "@fontsource/orbitron/700.css";
import "@fontsource/orbitron/900.css";
import "@fontsource/rajdhani/500.css";
import "@fontsource/rajdhani/600.css";
import "@fontsource/rajdhani/700.css";
import "@fontsource/share-tech-mono/400.css";
import "./index.css";

createRoot(document.getElementById("root")!).render(<App />);
