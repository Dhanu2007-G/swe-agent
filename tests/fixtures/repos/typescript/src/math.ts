export interface Vector {
  x: number;
  y: number;
}

export class Geometry {
  static distance(v1: Vector, v2: Vector): number {
    const dx = v1.x - v2.x;
    const dy = v1.y - v2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
}
