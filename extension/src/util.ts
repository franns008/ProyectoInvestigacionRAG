export function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}
