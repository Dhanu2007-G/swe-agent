import { Geometry, Vector } from "../src/math";

const v1: Vector = { x: 0, y: 0 };
const v2: Vector = { x: 3, y: 4 };
if (Geometry.distance(v1, v2) !== 5) {
  throw new Error("Test failed");
}
