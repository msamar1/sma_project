
1-Please install all required packages listed in requirements.txt
2- to test program :

    2-1 : function 1 ( test icm on action graphs with immunity ) enter following command:
        python main.py -fid 1 -r 500000 -c 5 -sc 0.2 -d 0 -ic 5 -dw False -de False
        fid : function_id  =1
        r : number of edges to be read from edge files
        c : number of initial active nodes
        sc: spreading coefficient which can be interpreted as the news attractiveness
        d : delay rounds between rumor and true news
        ic: number of initial true news spreader
        dw: draw graph in each round
        de: draw edge on graph

    2-2 : function 2 ( test icm on social graph ) enter following command:
        python main.py -fid 2 -r 500000 -c 5 -sc 0.3 -dw False
        fid : function_id  =2
        r : number of edges to be read from edge files
        c : number of initial active nodes
        sc: spreading coefficient which can be interpreted as the news attractiveness
        dw: draw graph in each round

    2-3 : function 3 ( test icm on action graphs ) enter following command:
        python main.py -fid 3 -r 500000 -c 5 -sc 0.3 -dw False
        fid : function_id  =2
        r : number of edges to be read from edge files
        c : number of initial active nodes
        sc: spreading coefficient which can be interpreted as the news attractiveness
        dw: draw graph in each round


