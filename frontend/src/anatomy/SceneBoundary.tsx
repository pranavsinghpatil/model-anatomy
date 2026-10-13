import { Component, type ReactNode } from "react";
import { Button } from "../components/ui";

export class SceneBoundary extends Component<
  { children: ReactNode; technical: () => void },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <div className="canvas-state" role="status">
        <strong>The illustration could not be drawn</strong>
        <p>
          The reported components and their evidence remain available in the
          technical graph.
        </p>
        <Button onClick={this.props.technical}>Use technical graph</Button>
      </div>
    ) : (
      this.props.children
    );
  }
}
