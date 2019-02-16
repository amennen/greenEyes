% this is going to load the right context description

function [cellOutput strOutput] = getContext(contextGroup)

cellOutput = {};
cellOutput{1} = 'You are about to hear a phone conversation.';
cellOutput{2} = 'It is late at night and the phone is ringing. On one end of the line is Arthur; Arthur just came home from a party. He left the party without finding his wife, Joanie.';
if contextGroup == 0
    % no context
    cellOutput{3} = 'One the other end is Lee, Arthur?s friend. He is at home and has just returned from the same party.';
        
elseif contextGroup == 1
    % paranoid context
    cellOutput{3} = 'As always, Arthur is paranoid, worrying that she might be having an affair, which is not true.';
    cellOutput{4} = 'On the other end is Lee, Arthur?s friend. He is at home with his girlfriend, Rose.  Lee and Rose have just returned from the same party, and are desperate to go to sleep.';
    cellOutput{5} = 'They do not know anything about Joanie?s whereabouts, and are tired of dealing with Arthur?s overreactions.';
    
elseif contextGroup == 2
    % cheating context
    cellOutput{3} = 'As always, Joanie was flirting with everybody at the party. Arthur is very upset.';
    cellOutput{4} = 'On the other end is Lee, Arthur?s friend.  He is at home with Joanie, Arthur?s wife. Lee and Joanie have just returned from the same party.  They have been having an affair for over a year now.';
    cellOutput{5} = 'They are thinking about the excuse Lee will use to calm Arthur this time.';
    
end

cellOutput{end+1} = 'All you need to do is remain still, listen, and pay close attention to the story.';
cellOutput{end+1} = '-- Please press your INDEX to begin once you understand these instructions. --';

strOutput = cellOutput{1};
for j = 2:length(cellOutput)-2
    strOutput = [strOutput ' ' cellOutput{j}];
end
    
end