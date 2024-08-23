function res=get_rot_axis(m)

fitfunc=@(x) vec(mat_rot_axis(x)-m);
a=optimoptions('lsqnonlin','Display','none');
res=lsqnonlin(fitfunc,[0,0,0],[],[],a);



function m=mat_rot_axis(an)
    m_IS = transform_matrix_rotation_arb_axis(0,0,an(1));
    m_RL = transform_matrix_rotation_arb_axis(90,0,an(2));
    m_AP = transform_matrix_rotation_arb_axis(90,90,an(3));
    
    %m=m_IS*m_RL*m_AP;
    %m=m_IS*m_AP*m_RL;
    %m=m_AP*m_RL*m_IS;
    %m=m_AP*m_IS*m_RL;
    %m=m_RL*m_IS*m_AP;
    m=m_AP*m_RL*m_IS;  % not sure which one is correct