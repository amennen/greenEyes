function [fileAvail specificFile] = GetSpecificClassOutputFile(classOutputDir,stationNum,usepython)

stationStr = num2str(stationNum);
if ~usepython
    specificFile = ['st_' stationStr '.mat'];
else
    specificFile = ['st_' stationStr '_py.txt'];
end
if exist(fullfile(classOutputDir,specificFile),'file');
    fileAvail = 1;
else
    fileAvail = 0;
end