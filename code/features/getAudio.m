function [yMat, Fs] = getAudio(filename,folder, h)
% Function to get 1hr of ELP audio file
% resulting signal matrix (samples x minute)

audiofile = fullfile(folder, filename);
info = audioinfo(audiofile);

samples = info.TotalSamples;
hr = round(samples/24);
min = round(hr/60);
ts = floor((min*2)-100);

for i = 1:30
    
    t0 = h*hr;
    tstart = t0+(ts*i);
    tend = tstart + ts;
    [y, Fs] = audioread(audiofile, [tstart tend]);
    yMat(:,i) = y; 

end

