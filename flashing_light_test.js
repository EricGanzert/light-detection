var execFile = require('child_process').execFile
var imgs_dir = "./light_flashing/img_captures/";

var child = execFile("./build/light-detection", [imgs_dir],
	function (error, stdout, stderr) {
		console.log("program output: \n" + stdout);
		var output = stdout;
		if (output == '0') {
			console.log("could not find flashing light");
		}
		else {
			console.log("flashing light found!");
		}
	});
