import type { GraphNode } from "../api/contract";
import { box, cylinder, type Face, type Vec3 } from "./solids";

type Point = Vec3;

const faceBudgetSegments = 20;

function addBox(
  faces: Face[],
  node: GraphNode,
  part: string,
  origin: Point,
  size: Point,
): void {
  faces.push(
    ...box(
      `${node.id}:${part}`,
      origin[0],
      origin[1],
      origin[2],
      size[0],
      size[1],
      size[2],
      node.id,
    ),
  );
}

function addCylinder(
  faces: Face[],
  node: GraphNode,
  part: string,
  origin: Point,
  radius: number,
  height: number,
): void {
  faces.push(
    ...cylinder(
      `${node.id}:${part}`,
      origin[0],
      origin[1],
      origin[2],
      radius,
      height,
      node.id,
      false,
      faceBudgetSegments,
    ),
  );
}

function stationBase(node: GraphNode, origin: Point): Face[] {
  const [x, y, z] = origin;
  const faces: Face[] = [];

  // A low mounting deck, four feet and paired side brackets attach every
  // mechanism to the same small bench footprint.
  addBox(faces, node, "mount-deck", [x - 31, y - 39, z + 4], [62, 78, 5]);
  for (const side of [-1, 1]) {
    for (const end of [-1, 1]) {
      addBox(
        faces,
        node,
        `foot-${side}-${end}`,
        [x + side * 21, y + end * 29, z],
        [8, 8, 5],
      );
    }
    addBox(
      faces,
      node,
      `bracket-${side}`,
      [x + side * 25, y - 33, z + 9],
      [5, 66, 4],
    );
  }
  // Two fine rails expose the join between the fixture and its top plate.
  addBox(faces, node, "carrier-plate", [x - 26, y - 28, z + 13], [52, 56, 4]);
  for (const end of [-1, 1]) {
    addBox(
      faces,
      node,
      `deck-seam-${end}`,
      [x - 30, y + end * 37, z + 9],
      [60, 1, 1],
    );
  }
  return faces;
}

function tokenCassette(
  node: GraphNode,
  origin: Point,
  separation: number,
  output: boolean,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const lift = separation * 28;
  const spread = separation * 2;
  if (!output) {
    const trayZ = z + 17 + lift;
    addBox(faces, node, "input-tray", [x - 24, y - 23, trayZ], [48, 42, 4]);
    for (const side of [-1, 1]) {
      addBox(
        faces,
        node,
        `tray-rail-${side}`,
        [x + side * (22 + spread), y - 23, trayZ + 4],
        [3, 42, 5],
      );
    }
    // Four visible inserts are a bounded illustration of a token tray, not a
    // token count reported by the model.
    for (let index = 0; index < 4; index += 1) {
      addBox(
        faces,
        node,
        `token-insert-${index}`,
        [
          x - 18 + index * 10 + spread * (index - 1.5),
          y - 12,
          trayZ + 4 + lift * 0.25,
        ],
        [7, 18, 4],
      );
    }
    return faces;
  }

  const trayZ = z + 17 + lift;
  addBox(
    faces,
    node,
    "output-collection-tray",
    [x - 24, y - 22, trayZ],
    [48, 38, 4],
  );
  addBox(
    faces,
    node,
    "output-backstop",
    [x - 24, y + 13, trayZ + 4],
    [48, 3, 7],
  );
  addBox(faces, node, "output-lip", [x - 24, y - 22, trayZ + 4], [48, 3, 3]);
  // The raised guides make this a receiving tray rather than another input rack.
  for (const side of [-1, 1]) {
    addBox(
      faces,
      node,
      `output-guide-${side}`,
      [x + side * 19, y - 16, trayZ + 4],
      [5, 26, 7],
    );
  }
  for (let index = 0; index < 3; index += 1) {
    addBox(
      faces,
      node,
      `output-insert-${index}`,
      [
        x - 15 + index * 10 + spread * (index - 1),
        y - 9,
        trayZ + 4 + lift * 0.25,
      ],
      [7, 14, 4],
    );
  }
  return faces;
}

function embeddingCartridge(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const spread = separation * 2;
  const cartridgeZ = z + 17 + separation * 32;
  addBox(
    faces,
    node,
    "lookup-cartridge",
    [x - 23, y - 24, cartridgeZ],
    [46, 48, 7],
  );
  addBox(
    faces,
    node,
    "contact-face",
    [x - 17, y - 17, cartridgeZ + 7],
    [34, 34, 2],
  );
  // A few large cells and two contact rails suggest a lookup cartridge without
  // encoding vocabulary size or any tensor dimensions into the drawing.
  for (let index = 0; index < 3; index += 1) {
    addBox(
      faces,
      node,
      `lookup-cell-${index}`,
      [x - 12 + index * 10 + spread * (index - 1), y - 10, cartridgeZ + 9],
      [7, 20, 3],
    );
  }
  for (const side of [-1, 1]) {
    addBox(
      faces,
      node,
      `contact-rail-${side}`,
      [x + side * 14, y - 18, cartridgeZ + 9],
      [2, 36, 2],
    );
  }
  return faces;
}

function decoderRack(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const spread = separation * 2;
  const lift = separation * 20;

  // A bounded sample, never an independent object for every reported instance.
  const representatives = Math.min(
    3,
    Math.max(1, Number.isFinite(node.repeat) ? Math.floor(node.repeat) : 1),
  );
  for (let index = 0; index < representatives; index += 1) {
    const offset = (index - (representatives - 1) / 2) * (13 + spread);
    addBox(
      faces,
      node,
      `block-cassette-${index}`,
      [x + offset - 6, y - 22, z + 17 + lift],
      [12, 44, 9],
    );
  }

  for (const side of [-1, 1])
    for (const end of [-1, 1])
      addBox(
        faces,
        node,
        `rack-post-${side}-${end}`,
        [x + side * 22 - 2, y + end * 22 - 2, z + 17],
        [4, 4, 44],
      );
  const coverZ = z + 61 + separation * 44;
  addBox(faces, node, "vented-cover", [x - 25, y - 25, coverZ], [50, 50, 6]);
  for (let index = 0; index < 2; index += 1) {
    addBox(
      faces,
      node,
      `cover-vent-${index}`,
      [x - 16 + index * 20, y - 25, coverZ + 6],
      [12, 50, 2],
    );
  }
  return faces;
}

function attentionManifold(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const spread = separation * 2;
  const manifoldZ = z + 17 + separation * 5;

  for (let index = 0; index < 3; index += 1) {
    addBox(
      faces,
      node,
      `manifold-channel-${index}`,
      [x - 23, y - 17 + index * (16 + spread), manifoldZ],
      [46, 4, 5],
    );
  }
  // A round mixing hub joins the three channels. Two cross-combs hold them in
  // a deliberate, inspectable fixture.
  addCylinder(faces, node, "mixing-hub", [x, y, manifoldZ + 4], 8, 6);
  for (const offset of [-8, 8]) {
    addBox(
      faces,
      node,
      `channel-comb-${offset}`,
      [x + offset - 1, y - 20, manifoldZ + 5],
      [2, 40, 3],
    );
  }
  return faces;
}

function projectionBanks(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const spread = separation * 1.5;
  const bankZ = z + 17 + separation * 4;
  for (const side of [-1, 1]) {
    const bankX = x + side * (15 + spread);
    addBox(
      faces,
      node,
      `projection-bank-${side}`,
      [bankX - 8, y - 18, bankZ],
      [16, 36, 10],
    );
    for (let fin = 0; fin < 2; fin += 1) {
      addBox(
        faces,
        node,
        `projection-fin-${side}-${fin}`,
        [bankX - 10, y - 12 + fin * 14, bankZ + 10],
        [20, 5, 6],
      );
    }
  }
  addBox(faces, node, "bank-header", [x - 24, y + 22, bankZ], [48, 5, 4]);
  return faces;
}

function normalizationDrum(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const drumZ = z + 17;
  addCylinder(faces, node, "calibration-drum", [x, y, drumZ], 14, 14);
  const dialZ = drumZ + 14 + separation * 24;
  addCylinder(faces, node, "lifted-dial", [x, y, dialZ], 11, 4);
  for (let index = 0; index < 3; index += 1) {
    addBox(
      faces,
      node,
      `calibration-mark-${index}`,
      [x - 8 + index * 7, y - 1, dialZ + 4],
      [2, 6, 2],
    );
  }
  addBox(faces, node, "dial-pointer", [x - 1, y + 3, dialZ + 4], [2, 8, 2]);
  return faces;
}

function summingJunction(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const spread = separation * 1.5;
  const junctionZ = z + 17;

  // Two receiving channels meet the central block; one channel leaves at a
  // right angle. These are parts of the metaphor, not graph edges or port data.
  addBox(
    faces,
    node,
    "receiving-channel-left",
    [x - 24, y - 2 - spread, junctionZ],
    [19, 4, 4],
  );
  addBox(
    faces,
    node,
    "receiving-channel-right",
    [x + 5, y - 2 + spread, junctionZ],
    [19, 4, 4],
  );
  addBox(
    faces,
    node,
    "junction-block",
    [x - 6, y - 6, junctionZ - 1],
    [12, 12, 9],
  );
  addBox(
    faces,
    node,
    "single-outlet-channel",
    [x - 2, y + 5, junctionZ],
    [4, 20, 4],
  );

  // At rest the cap rests on the junction block; separation lifts the cover,
  // leaving the two receiving channels and the outlet visibly attached below.
  addBox(
    faces,
    node,
    "junction-cover",
    [x - 10, y - 10, junctionZ + 8 + separation * 12],
    [20, 20, 3],
  );
  return faces;
}

function dropoutScreen(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const screenZ = z + 13 + separation * 10;
  const spread = separation * 1.5;

  // The outer rails and three fixed cross-slots make a removable perforated
  // screen. Its illustrative openings never flicker or encode a runtime mask.
  addBox(
    faces,
    node,
    "screen-frame-left",
    [x - 25 - spread, y - 24, screenZ],
    [3, 48, 4],
  );
  addBox(
    faces,
    node,
    "screen-frame-right",
    [x + 22 + spread, y - 24, screenZ],
    [3, 48, 4],
  );
  addBox(
    faces,
    node,
    "screen-frame-front",
    [x - 22, y - 24, screenZ],
    [44, 3, 4],
  );
  addBox(
    faces,
    node,
    "screen-frame-back",
    [x - 22, y + 21, screenZ],
    [44, 3, 4],
  );
  for (let index = 0; index < 3; index += 1) {
    addBox(
      faces,
      node,
      `screen-slot-bar-${index}`,
      [x - 11 + index * 11 + spread * (index - 1), y - 21, screenZ],
      [2, 42, 3],
    );
  }
  return faces;
}

function rotaryWheel(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const wheelZ = z + 17;
  addCylinder(faces, node, "phase-wheel", [x, y, wheelZ], 17, 6);
  addBox(faces, node, "wheel-spoke-x", [x - 15, y - 2, wheelZ + 6], [30, 4, 3]);
  addBox(faces, node, "wheel-spoke-y", [x - 2, y - 15, wheelZ + 6], [4, 30, 3]);
  addCylinder(
    faces,
    node,
    "raised-wheel-hub",
    [x, y, wheelZ + 9 + separation * 8],
    5,
    6,
  );
  addBox(
    faces,
    node,
    "phase-pointer",
    [x + 3, y + 3, wheelZ + 9 + separation * 8],
    [15, 3, 3],
  );
  return faces;
}

function sealedCase(
  node: GraphNode,
  origin: Point,
  separation: number,
): Face[] {
  const [x, y, z] = origin;
  const faces = stationBase(node, origin);
  const caseZ = z + 17;
  const lidZ = caseZ + 25 + separation * 36;
  addBox(faces, node, "sealed-housing", [x - 22, y - 25, caseZ], [44, 50, 25]);
  addBox(faces, node, "sealed-lid", [x - 24, y - 27, lidZ], [48, 54, 5]);
  // External seam bars and lock blocks keep the unknown mechanism visibly shut.
  for (const side of [-1, 1]) {
    addBox(
      faces,
      node,
      `case-seam-${side}`,
      [x + side * 22, y - 22, caseZ + 8],
      [2, 44, 2],
    );
  }
  for (const end of [-1, 1]) {
    addBox(
      faces,
      node,
      `lid-lock-${end}`,
      [x - 4, y + end * 25, lidZ - 2],
      [8, 4, 5],
    );
  }
  return faces;
}

const inputKinds = new Set(["input"]);
const outputKinds = new Set(["output"]);
const embeddingKinds = new Set([
  "embedding",
  "embeddings",
  "token_embedding",
  "position_embedding",
  "patch_embedding",
]);
const decoderKinds = new Set([
  "decoder",
  "decoder_block",
  "decoder_layer",
  "transformer_block",
]);
const attentionKinds = new Set([
  "attention",
  "self_attention",
  "cross_attention",
  "attention_core",
]);
const projectionKinds = new Set([
  "mlp",
  "feed_forward",
  "gated_mlp",
  "projection",
  "projection_block",
  "linear",
  "head",
  "lm_head",
  "classification_head",
]);
const normalizationKinds = new Set([
  "layer_norm",
  "rms_norm",
  "batch_norm",
  "normalization",
  "norm",
]);
const rotaryKinds = new Set(["rotary_position_embedding", "rotary_embedding"]);
const summingKinds = new Set(["add", "residual"]);
const dropoutKinds = new Set(["dropout"]);

/**
 * Build one bounded physical metaphor from the node's recorded kind. Unknown
 * kinds remain sealed; labels, tensor names and neighboring nodes never choose
 * the mechanism.
 */
export function mechanism(
  node: GraphNode,
  origin: Vec3,
  separation: number,
): Face[] {
  const spread = Math.max(
    0,
    Math.min(1, Number.isFinite(separation) ? separation : 0),
  );
  const kind = node.kind.toLowerCase();
  if (inputKinds.has(kind)) return tokenCassette(node, origin, spread, false);
  if (outputKinds.has(kind)) return tokenCassette(node, origin, spread, true);
  if (embeddingKinds.has(kind)) return embeddingCartridge(node, origin, spread);
  if (decoderKinds.has(kind)) return decoderRack(node, origin, spread);
  if (attentionKinds.has(kind)) return attentionManifold(node, origin, spread);
  if (projectionKinds.has(kind)) return projectionBanks(node, origin, spread);
  if (normalizationKinds.has(kind))
    return normalizationDrum(node, origin, spread);
  if (rotaryKinds.has(kind)) return rotaryWheel(node, origin, spread);
  if (summingKinds.has(kind)) return summingJunction(node, origin, spread);
  if (dropoutKinds.has(kind)) return dropoutScreen(node, origin, spread);
  return sealedCase(node, origin, spread);
}
