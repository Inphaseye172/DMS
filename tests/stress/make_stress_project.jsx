// Builds the aerender stress project: one 3840x2160 comp per model x mapping x tiles combination,
// each with an animated 4K still (content changes every frame) and the DMS effect applied.
// Run by tests/stress/aerender_stress.ps1 via:  AfterFX.exe -noui -r make_stress_project.jsx
// Inputs (environment): DMS_STRESS_PROJECT (output .aep), DMS_STRESS_FOOTAGE (4K image),
//                       DMS_STRESS_FRAMES (default 500), DMS_STRESS_MODELS (comma list of popup indices, default "1")
(function () {
    function env(name, dflt) { var v = $.getenv(name); return (v === null || v === "") ? dflt : v; }
    var projectPath = env("DMS_STRESS_PROJECT", "");
    var footagePath = env("DMS_STRESS_FOOTAGE", "");
    var frames = parseInt(env("DMS_STRESS_FRAMES", "500"), 10);
    var models = env("DMS_STRESS_MODELS", "1").split(",");
    var log = new File(projectPath + ".log");
    log.open("w");
    try {
        if (!projectPath || !footagePath) { throw new Error("DMS_STRESS_PROJECT and DMS_STRESS_FOOTAGE must be set"); }
        app.beginSuppressDialogs();
        app.newProject();
        var proj = app.project;
        var footage = proj.importFile(new ImportOptions(new File(footagePath)));
        var fps = 24;
        var mappings = [["raw", 1], ["normalize", 2], ["guided", 3]];
        var tiles = [["t1", 2], ["t4", 3], ["t16", 5]];  // Computation Tiles popup: Auto=1, 1=2, 4=3, 9=4, 16=5
        for (var m = 0; m < models.length; m++) {
            for (var a = 0; a < mappings.length; a++) {
                for (var t = 0; t < tiles.length; t++) {
                    var name = "DMS_m" + models[m] + "_" + mappings[a][0] + "_" + tiles[t][0];
                    var comp = proj.items.addComp(name, 3840, 2160, 1, frames / fps, fps);
                    var layer = comp.layers.add(footage);
                    layer.outPoint = comp.duration;
                    var tr = layer.property("ADBE Transform Group");
                    var pos = tr.property("ADBE Position");
                    pos.setValueAtTime(0, [1920, 1080]);
                    pos.setValueAtTime(comp.duration, [2150, 1200]);
                    var scale = tr.property("ADBE Scale");
                    scale.setValueAtTime(0, [100, 100]);
                    scale.setValueAtTime(comp.duration, [118, 118]);
                    var fx = layer.property("ADBE Effect Parade").addProperty("DMS DepthMap AI");
                    if (!fx) { throw new Error("effect 'DMS DepthMap AI' not found - is the plugin installed?"); }
                    fx.property("Model").setValue(parseInt(models[m], 10));
                    fx.property("Mapping Mode").setValue(mappings[a][1]);
                    fx.property("Computation Tiles").setValue(tiles[t][1]);
                    if (mappings[a][0] === "guided") {
                        fx.property("Near Guide Point").setValue([1920, 2000]);
                        fx.property("Far Guide Point").setValue([1920, 300]);
                    }
                    log.writeln("created " + name);
                }
            }
        }
        proj.save(new File(projectPath));
        log.writeln("saved " + projectPath);
    } catch (e) {
        log.writeln("ERROR: " + e.toString() + " line " + e.line);
    } finally {
        log.close();
        app.endSuppressDialogs(false);
        app.quit();
    }
})();
