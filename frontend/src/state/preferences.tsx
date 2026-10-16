import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
export type Mode = "learn" | "research";
type Theme = "light" | "dark" | "system";
type Settings = {
  mode: Mode;
  theme: Theme;
  setMode: (mode: Mode) => void;
  setTheme: (theme: Theme) => void;
};
const Context = createContext<Settings | null>(null);
function read(key: string) {
  try {
    return localStorage.getItem(`anatomy-${key}`);
  } catch {
    return null;
  }
}
export function Preferences({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>(() =>
    read("mode") === "learn" ? "learn" : "research",
  );
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = read("theme");
    return saved === "light" || saved === "dark" || saved === "system"
      ? saved
      : "light";
  });
  useEffect(() => {
    try {
      localStorage.setItem("anatomy-mode", mode);
      localStorage.setItem("anatomy-theme", theme);
    } catch {
      /* Storage may be blocked. */
    }
    const media = matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      const resolved =
        theme === "system" ? (media.matches ? "dark" : "light") : theme;
      document.documentElement.dataset.theme = resolved;
      document.documentElement.style.colorScheme = resolved;
    };
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [theme, mode]);
  return (
    <Context.Provider value={{ mode, theme, setMode, setTheme }}>
      {children}
    </Context.Provider>
  );
}
export function usePreferences() {
  const value = useContext(Context);
  if (!value) throw new Error("Preferences provider missing");
  return value;
}
