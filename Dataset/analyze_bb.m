for i=0:49
if -eval([sprintf('pedestrian_positions.actor_%d(701,2)',i)])>-170 && -eval([sprintf('pedestrian_positions.actor_%d(701,2)',i)]) < -140
    i
    [- eval([sprintf('pedestrian_positions.actor_%d(701,2)',i)]), eval([sprintf('pedestrian_positions.actor_%d(701,1)',i)])]
end
% [- eval([sprintf('pedestrian_positions.actor_%d(701,2)',i)]), eval([sprintf('pedestrian_positions.actor_%d(701,1)',i)])]
end