import { lazy, Suspense, useEffect } from "react";
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import { Preferences } from "./state/preferences";
import { Start } from "./pages/Entry";
import { Loading } from "./components/Common";
const Workspace = lazy(() => import("./pages/Workspace"));
const Catalog = lazy(() => import("./pages/Catalog"));
function RouteFocus() {
  const { pathname } = useLocation();
  useEffect(() => {
    document.querySelector<HTMLElement>("main")?.focus();
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}
function Home() {
  const legacy = new URLSearchParams(useLocation().search).get("analysis");
  return legacy ? (
    <Navigate replace to={`/app/analyses/${encodeURIComponent(legacy)}`} />
  ) : (
    <Start />
  );
}
export function App() {
  return (
    <Preferences>
      <BrowserRouter>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <RouteFocus />
        <Suspense
          fallback={
            <main id="main" tabIndex={-1} className="page-state">
              <Loading text="Opening workspace" />
            </main>
          }
        >
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/app" element={<Start />} />
            <Route path="/catalog" element={<Catalog />} />
            <Route
              path="/app/catalog"
              element={<Navigate replace to="/catalog" />}
            />
            <Route path="/app/analyses/:id" element={<Workspace />} />
            <Route
              path="*"
              element={
                <main id="main" tabIndex={-1} className="page-state">
                  <h1>Page not found</h1>
                  <a href="/app">Start an inspection</a>
                </main>
              }
            />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </Preferences>
  );
}
