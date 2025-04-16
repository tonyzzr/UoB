import DatasetExplorer from "@/components/DatasetExplorer"; // Using alias @/

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-2xl font-bold mb-6">Ultrasound Data Explorer</h1>
      </div>

      <div className="w-full max-w-5xl mt-10">
        <DatasetExplorer />
      </div>

      {/* You can add more standard Next.js welcome elements or other components here later */}
    </main>
  );
} 