import tls from "node:tls";
import { trustSystemCertificates, type TlsCaApi } from "@/lib/systemCa";

const added = trustSystemCertificates(tls as unknown as TlsCaApi);
if (added > 0) console.log(`[tls] trusting ${added} certificates from the OS store for server-side HTTPS`);
