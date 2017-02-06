function animate_frames(frames,fps,fskip,c)

if nargin==3
    c='b';
elseif nargin==2
    fskip=1;
    c='b'
elseif nargin==1
        
    fps=60; 
    fskip=1;
    c='b'
end


elem=size(frames,1)/3;



    if size(frames,1)==1
            f_N=size(frames,3)
            for i=1:fskip:f_N
            clf
            
             plot3(frames(1,1:3:end,i),frames(1,2:3:end,i),frames(1,3:3:end,i),strcat('.',c));
%              wp(1:2:end)=1:elem;
%              wp(2:2:end)=parents(2:end);
%              
%              plot3(frames(1,wp,i),frames(1,wp,i),frames(1,wp,i),strcat('-',c))

               view(0,90);
               title(num2str(i));
                drawnow;
                pause(1/fps)

            end
        else
            f_N=size(frames,2)
            t=0.2;
xl=[min(min(frames(1:3:end,:)))-t max(max(frames(1:3:end,:)))+t];
yl=[min(min(frames(2:3:end,:)))-t max(max(frames(2:3:end,:)))+t];
zl=[min(min(frames(3:3:end,:)))-t max(max(frames(3:3:end,:)))+t];
            for i=1:fskip:f_N
            clf
            plot3(frames(1:3:end,i),frames(2:3:end,i),frames(3:3:end,i),strcat('.',c));
            
            
             xlim(xl);
             ylim(yl);
             zlim(zl);


               view(0,90);
               title(num2str(i));
                drawnow;
                pause(1/fps)
                
                im = frame2im(getframe(1));
                [imind,cm] = rgb2ind(im,256);
                outfile = 'sinewave.gif';
 
    % On the first loop, create the file. In subsequent loops, append.
%     if i==1
%         imwrite(imind,cm,outfile,'gif','DelayTime',0,'loopcount',inf);
%     else
%         imwrite(imind,cm,outfile,'gif','DelayTime',0,'writemode','append');
%     end
%                 
% 
            end
%         
%     end


end