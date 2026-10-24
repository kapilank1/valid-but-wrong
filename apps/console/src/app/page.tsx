import { ConsoleApp } from "@/components/ConsoleApp";
import { EngineProvider } from "@/components/EngineProvider";

export default function Page() {
  return (
    <EngineProvider>
      <ConsoleApp />
    </EngineProvider>
  );
}
