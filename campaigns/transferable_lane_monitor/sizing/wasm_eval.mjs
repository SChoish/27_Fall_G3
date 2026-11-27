// Optional ngspice CLI bridge for the pinned Emscripten build. Host reads are explicit.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [modulePath, modelPath, netlistPath] = process.argv.slice(2);
const create = (await import(pathToFileURL(modulePath))).default;
const sim = await create({noInitialRun:true, print: s=>console.log(s), printErr:s=>console.error(s)});
sim.FS.writeFile("/proc/meminfo", "MemTotal:        2097152 kB\nMemFree:         1048576 kB\nMemAvailable:    1048576 kB\n");
sim.FS.writeFile('/model.spice', fs.readFileSync(modelPath));
sim.FS.writeFile('/design.cir', fs.readFileSync(netlistPath));
const args = ['ngspice','-b','/design.cir'];
const argv = sim.stackAlloc((args.length+1)*4);
for(let i=0;i<args.length;i++){
  const ptr = sim.stringToUTF8OnStack(args[i]);
  sim.setValue(argv+4*i,ptr,'i32');
}
sim.setValue(argv+4*args.length,0,'i32');
try {
  process.exitCode = sim._main(args.length,argv);
} catch (error) {
  if (error?.name !== "ExitStatus") throw error;
  process.exitCode = error.status;
}
