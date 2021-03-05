function [datatable,nfiles] = getData(folder)
% Function 
% detailed description

% get 
datafiles = dir(fullfile(folder, '*.wav'));

% number of audio files in data
nfiles = length(datafiles);

% get file names into datatable.filename
for i = 1:nfiles
    filename(i,:) = string(datafiles(i).name);
end

datatable = table(filename);

end

