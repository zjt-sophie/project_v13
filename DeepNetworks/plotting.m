% figure % create new figure
% for i= 1: 25
%  subplot(5,5,i) % first subplot
% %  plot_sum(i)=sum(Feature_DTest(i,:))
%  plot(test_sae1FeaturesL3(i,:),'Color',[0.4 0.2 0.6]);
%  title('unit');
% end

% figure
% bar (plot_sum);
%  for i= 1: 13
%   subplot(5,3,i) % first subplot
%  %  plot_sum(i)=sum(Feature_DTest(i,:))
%   plot(test_sae2FeaturesL3(i,:),'Color',[0.4 0.2 0.6]);
%   title('unit');
%  end

%  for i= 1: 13
%   subplot(5,5,i) % first subplot
%  %  plot_sum(i)=sum(Feature_DTest(i,:))
%   plot( sae2Features(i,:),'Color',[0.4 0.2 0.6]);
%   title('unit');
%  end

 

 
%  for i= 1: 25
%   subplot(5,5,i) % first subplot
% %     plot_sum(i)=sum(Feature_DTest(i,:))
%   plot( sae1Features(i,:),'Color',[0.4 0.2 0.6]);
%   title('unit');
%  end

%  figure
%  for j = 1 : 4
%     subplot(4,4,j);   
%     plot( Action_RTest (j,:),'Color',[0.4 0.2 0.6]);
%  end
%  mean(Action_RTest (1,:))
%  mean(Action_RTest (2,:))
%  mean(Action_RTest (3,:))
%  mean(Action_RTest (4,:))

W = R_W1'*W1L1;
display_network(W', 12);
                                 
                 print -djpeg weights2.jpg   % save the visualization to a file