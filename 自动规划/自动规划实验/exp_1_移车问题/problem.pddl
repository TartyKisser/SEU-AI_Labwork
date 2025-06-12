(define (problem vehicle_arrangement) (:domain vehicle_parking)
    (:objects 
        v1 v2 v3 v4 - vehicle
        g1 g2 g3 - garage
        s1 s2 - slot
    )
    (:init
        (top_slot s2)
        (adjacent_slot s1 s2)
        (located v1 g1 s2)
        (occupied g1 s2)
        (located v2 g2 s1)
        (occupied g2 s1)
        (located v3 g2 s2)
        (occupied g2 s2)
        (located v4 g3 s1)
        (occupied g3 s1)
    )
    (:goal (and
            (located v1 g2 s2)
            (occupied g2 s2)
            (located v2 g1 s1)
            (occupied g1 s1)
            (located v3 g3 s2)
            (occupied g3 s2)
            (located v4 g1 s2)
            (occupied g1 s2)
        )
    )
)