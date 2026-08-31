import { GalleryWorkbench } from "@/components/gallery-workbench";
import { loadGallery } from "@/lib/data";

export default async function Home() {
  const gallery = await loadGallery();
  return <GalleryWorkbench initialData={gallery} />;
}
