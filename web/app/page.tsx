import { GalleryWorkbench } from "@/components/gallery-workbench";
import { loadGallery } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function Home() {
  const gallery = await loadGallery();
  return <GalleryWorkbench initialData={gallery} />;
}
