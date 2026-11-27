// Each deck receives new Emscripten memory, FS and simulator state.
import fs from "node:fs";
import readline from "node:readline";
import {pathToFileURL} from "node:url";
const [modulePath, modelPath] = process.argv.slice(2);
const create = (await import(pathToFileURL(modulePath))).default;
const source = fs.readFileSync(modulePath,"utf8");
const payload = source.match(/function findWasmBinary\(\)\{return base64Decode\("([A-Za-z0-9+/=]+)"\)/);
if (!payload) throw Error("The selected ngspice build has no embedded wasm payload");
const compiled = await WebAssembly.compile(Buffer.from(payload[1],"base64"));
const model = fs.readFileSync(modelPath);
for await (const line of readline.createInterface({input:process.stdin})) {
    const output = [];
    let status = 0;
    try {
        const sim = await create({
            noInitialRun: true,
            print: s => output.push(s),
            printErr: s => output.push(s),
            instantiateWasm: (imports,ready) => ready(new WebAssembly.Instance(compiled,imports),compiled)
        });
        sim.FS.writeFile("/proc/meminfo","MemTotal:        2097152 kB\nMemFree:         1048576 kB\nMemAvailable:    1048576 kB\n");
        sim.FS.writeFile("/model.spice",model);
        sim.FS.writeFile("/design.cir",JSON.parse(line));
        const args=["ngspice","-b","/design.cir"];
        const argv=sim.stackAlloc(4*(args.length+1));
        args.forEach((s,i)=>sim.setValue(argv+4*i,sim.stringToUTF8OnStack(s),"i32"));
        sim.setValue(argv+4*args.length,0,"i32");
        status=sim._main(args.length,argv);
    } catch (e) {
        if(e?.name!=="ExitStatus") throw e;
        status=e.status;
    }
    process.stdout.write(JSON.stringify([status,output.join("\n")])+"\n");
}
