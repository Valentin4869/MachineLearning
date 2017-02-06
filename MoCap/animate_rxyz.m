function animate_rxyz(rots,parents, offsets,fps,fskip)

if nargin==3
    fps=60;
    fskip=1;
    
      
    
elseif nargin==1

    try
    parents=load(strcat('Data/Static/','parents.mat'));
    parents=parents.parents;
    offsets=load(strcat('Data/Static/','offsets.mat'));
    offsets=offsets.offsets;
    catch
       fprintf('\nCan''t find parents and offsets!\n Pass them as arguments if they are not available on disk.'); 
    end
    fps=60; 
    fskip=1;
   
end


frames=rxyz2dxyz(rots,parents,offsets);

animate_frames(frames,fps,fskip);

end