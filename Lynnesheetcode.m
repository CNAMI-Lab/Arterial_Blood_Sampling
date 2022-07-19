contents = readlines(fullfile("NHP_timings_ASEM_AZAN.csv"));
%contents1 = readlines(fullfile(getenv('MATLAB'),'Tracer','Date','Subject ID:','Subject wt (kg)','Dose (mCi):', 'Start Inj (clock):,','Sample', 'Timepoint','Clock Time start','Clock Time stop','Notes','NHP_timings_ASEM_AZAN.csv'));
size(contents)
contents(1)
data = csvread('DOWN.csv');
data(1:(size(contents)),1)
%find label/column, identify entries: read entries that correspond to
%labels
%regexpi(contents(1), '1:15:00')

%only 1 entry %need 'names'??
Tracer=regexpi(contents(1), '(?<Tracer>\S+*\d+''names');
Date=regexpi(contents(1), '(?<Month>\d+)/(?<Day>\d+)/(?<Year>\d+)|(?<day>\d+).(?<month>\d+).(?<year>\d+)',...
    'names');
SubjectID=regexpi(contents(1), '(?<SubjectID>\w+''names');
SubjectWt=regexpi(contents(1), '(?<SubjectWt>(\d+).(\d+))''names');
Dose=regexpi(contents(1), '(?<Dose>(\d+).(\d+))''names');
Volume=regexpi(contents(1), '(?<Volume>(\d+).(\d+))''names');
%time parameters here
StartInj=regexpi(contents(1), '(?<Hour>\d+):(?<Minute>\d+):(?<Second>\d+)''names');
%ensure time has passed
StopInj=regexpi(contents(1), '(?<Hour>\d+):(?<Minute>\d+):(?<Second>\d+)''names');


SampleNum=regexpi(contents(1), '(?<SampleNum>\d+)|[HPLC]');
data = csvread('DOWN.csv');
data(1:(size(contents)),1);
result = cell(size(input, 1), 2);
for row = 1 : size(contents, 1)
    tokens = regexp(contents{row}, '(.*)=(.*)', 'tokens');
    if ~isempty(tokens)
        result(row, :) = tokens{1};
  %Sample = csvread("NHP_timings_ASEM_AZAN.csv", 0, 1, [0 1 "SampleNum"-1 1 ] );
%while ((SampleNum>=1)||(isnan(SampleNum))) %multiple entries
    %HPLC=regexpi(contents(1), '(?<HPLC>');
    TimePoint=regexpi(contents(1), '(?<Hour>\d+):(?<Minute>\d+):(?<Second>\d+)''SampleNum');
    ClockStart=regexpi(contents(1), '(?<Hour>\d+):(?<Minute>\d+):(?<Second>\d+)''SampleNum');
    ClockStop=regexpi(contents(1), '(?<Hour>\d+):(?<Minute>\d+):(?<Second>\d+)''SampleNum');
    Notes=regexpi(contents(1), '(?<Notes>\w+');
     end
end 
%end
%fprintf('%s\n', re)
%save/store data