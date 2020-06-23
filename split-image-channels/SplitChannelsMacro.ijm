

function splitChannels(input, output, filename) {
	open(input + filename);
	run("Split Channels");
	selectWindow("C2-"+filename);
	saveAs("Tiff", output + "rna/raw-data/" + filename);
	close();
	selectWindow("C1-"+filename);
	saveAs("Tiff", output + "centrosomes/raw-data/" + filename);
	close();
	selectWindow("C3-"+filename);
	saveAs("Tiff", output + "nuclei/raw-data/" + filename);
	close();
}

// Define input and output directory paths

directory = "/Users/pearlryder/data/resubmission-data/data/"
output = "/Users/pearlryder/data/resubmission-data/data/segmentation/"

input = "staging/"


input_dir = directory + input
output_dir = output

// iterate over files in input directory and perform SplitChannels on each file

setBatchMode(true);

list = getFileList(input_dir);
for (i = 0; i < list.length; i++)
	splitChannels(input_dir, output_dir, list[i]);


setBatchMode(false);