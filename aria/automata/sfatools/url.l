
SCHEME      ([a-zA-Z][a-zA-Z0-9+.-]*)
PORT         :[0-9]+
PATH         \/[a-zA-Z0-9_\-\.%&]*

%%
{SCHEME}:\/\/[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)+({PORT})?({PATH})* {ACCEPT;}
%%

