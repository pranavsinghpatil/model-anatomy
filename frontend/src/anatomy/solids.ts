export type Vec3 = [number, number, number];

export type FaceTone = "top" | "front" | "side" | "dark" | "accent";

export type Face = {
  id: string;
  points: Vec3[];
  tone: FaceTone;
  nodeId?: string;
};

const face = (
  id: string,
  points: Vec3[],
  tone: FaceTone,
  nodeId?: string,
): Face => ({
  id,
  points,
  tone,
  ...(nodeId ? { nodeId } : {}),
});

/**
 * A box in world coordinates. x/y/z are the lower corner; w/d/h extend along
 * positive X/Y/Z. These dimensions belong to the illustration, not the model.
 */
export function box(
  id: string,
  x: number,
  y: number,
  z: number,
  w: number,
  d: number,
  h: number,
  nodeId?: string,
  accent = false,
): Face[] {
  const x0 = x;
  const x1 = x + w;
  const y0 = y;
  const y1 = y + d;
  const z0 = z;
  const z1 = z + h;
  return [
    face(
      `${id}:top`,
      [
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
      ],
      accent ? "accent" : "top",
      nodeId,
    ),
    face(
      `${id}:front`,
      [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y0, z1],
        [x0, y0, z1],
      ],
      "front",
      nodeId,
    ),
    face(
      `${id}:right`,
      [
        [x1, y0, z0],
        [x1, y1, z0],
        [x1, y1, z1],
        [x1, y0, z1],
      ],
      "side",
      nodeId,
    ),
    face(
      `${id}:back`,
      [
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y1, z1],
        [x1, y1, z1],
      ],
      "dark",
      nodeId,
    ),
    face(
      `${id}:left`,
      [
        [x0, y1, z0],
        [x0, y0, z0],
        [x0, y0, z1],
        [x0, y1, z1],
      ],
      "side",
      nodeId,
    ),
    face(
      `${id}:bottom`,
      [
        [x0, y1, z0],
        [x1, y1, z0],
        [x1, y0, z0],
        [x0, y0, z0],
      ],
      "dark",
      nodeId,
    ),
  ];
}

/** A vertical faceted cylinder, centered on x/y and based at z. */
export function cylinder(
  id: string,
  x: number,
  y: number,
  z: number,
  r: number,
  h: number,
  nodeId?: string,
  accent = false,
  segments = 20,
): Face[] {
  const count = Math.max(6, Math.min(32, Math.floor(segments) || 20));
  const lower: Vec3[] = [];
  const upper: Vec3[] = [];
  for (let index = 0; index < count; index += 1) {
    const angle = (Math.PI * 2 * index) / count;
    const px = x + Math.cos(angle) * r;
    const py = y + Math.sin(angle) * r;
    lower.push([px, py, z]);
    upper.push([px, py, z + h]);
  }

  const faces: Face[] = [
    face(`${id}:top`, upper, accent ? "accent" : "top", nodeId),
    face(`${id}:bottom`, [...lower].reverse(), "dark", nodeId),
  ];
  for (let index = 0; index < count; index += 1) {
    const next = (index + 1) % count;
    const tone: FaceTone =
      Math.sin(((index + 0.5) * Math.PI * 2) / count) < -0.35
        ? "front"
        : "side";
    faces.push(
      face(
        `${id}:wall:${index}`,
        [lower[index], lower[next], upper[next], upper[index]],
        tone,
        nodeId,
      ),
    );
  }
  return faces;
}
