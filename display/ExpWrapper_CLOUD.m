%toml_file = '/Volumes/norman/amennen/github/brainiak/rt-cloud/projects/greenEyes/conf/greenEyes_organized.toml';
dbstop if error; % for audio problems on linux
toml_file = '/Data1/code/rt-cloud/projects/greenEyes/conf/greenEyes_organized.toml';
runNum = 1;

RealTimeGreenEyesDisplay_CLOUD(toml_file, runNum)