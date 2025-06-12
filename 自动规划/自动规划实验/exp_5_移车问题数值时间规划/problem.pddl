(define (problem vehicle-transport-problem)
    (:domain vehicle-transport)
    
    (:objects
        v1 v2 - vehicle
        l1 l2 l3 l4 l5 - location
        p1 p2 p3 p4 - package
        r1 r2 r3 r4 r5 r6 - road
    )
    
    (:init
        ; 车辆初始位置和状态
        (at v1 l1) (assign (fuel-level v1) 80) (has-fuel v1)
        (at v2 l5) (assign (fuel-level v2) 60) (has-fuel v2)
        
        ; 包裹初始位置和目的地
        (package-at p1 l1) (destination p1 l3)
        (package-at p2 l2) (destination p2 l4)
        (package-at p3 l3) (destination p3 l5)
        (package-at p4 l4) (destination p4 l1)
        
        ; 道路连接和状况
        (connected r1 l1 l2) (assign (distance r1) 10) (road-condition r1 good)
        (connected r2 l2 l3) (assign (distance r2) 15) (road-condition r2 bad)
        (connected r3 l3 l4) (assign (distance r3) 8)  (road-condition r3 good)
        (connected r4 l4 l5) (assign (distance r4) 12) (road-condition r4 good)
        (connected r5 l1 l5) (assign (distance r5) 20) (road-condition r5 bad)
        (connected r6 l2 l4) (assign (distance r6) 5)  (road-condition r6 good)
        
        ; 初始时间和成本
        (assign (total-time) 0)
        (assign (travel-time v1) 0)
        (assign (travel-time v2) 0)
        (assign (maintenance-cost v1) 0)
        (assign (maintenance-cost v2) 0)
        (assign (delivery-delay p1) 0)
        (assign (delivery-delay p2) 0)
        (assign (delivery-delay p3) 0)
        (assign (delivery-delay p4) 0)
        
        ; 动态包裹到达（时序初始文字）
        (= (at 5) (package-at p2 l2))  ; 包裹2在时间点5到达
        (= (at 10) (package-at p4 l4)) ; 包裹4在时间点10到达
    )
    
    (:goal (and
        ; 所有包裹到达目的地
        (package-at p1 l3)
        (package-at p2 l4)
        (package-at p3 l5)
        (package-at p4 l1)
        
        ; 车辆返回基地（可选）
        (at v1 l1)
        (at v2 l5)
    ))
    
    (:metric minimize (+ (* 2 (total-time))            ; 总时间权重
                         (* 1 (maintenance-cost v1))    ; 车辆1维护成本权重
                         (* 1 (maintenance-cost v2))    ; 车辆2维护成本权重
                         (* 0.5 (delivery-delay p1))    ; 包裹1延迟权重
                         (* 0.5 (delivery-delay p2))    ; 包裹2延迟权重
                         (* 0.5 (delivery-delay p3))    ; 包裹3延迟权重
                         (* 0.5 (delivery-delay p4)))))  ; 包裹4延迟权重
)    