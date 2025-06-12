(define (domain vehicle-transport)
    (:requirements :typing :fluents :durative-actions :timed-initial-literals :numeric-fluents)
    
    (:types 
        vehicle - object
        location - object
        package - object
        road - object
    )
    
    (:predicates
        (at ?v - vehicle ?l - location)            ; 车辆v在位置l
        (loaded ?v - vehicle ?p - package)        ; 车辆v装载了包裹p
        (package-at ?p - package ?l - location)   ; 包裹p在位置l
        (destination ?p - package ?l - location)  ; 包裹p的目的地是l
        (connected ?r - road ?l1 ?l2 - location)  ; 道路r连接位置l1和l2
        (road-condition ?r - road ?c - {good bad}) ; 道路r的状况
        (has-fuel ?v - vehicle)                   ; 车辆v有燃料
    )
    
    (:functions
        (total-time)                              ; 总时间
        (fuel-level ?v - vehicle)                 ; 车辆v的燃料水平
        (distance ?r - road)                      ; 道路r的距离
        (delivery-delay ?p - package)             ; 包裹p的交付延迟
        (travel-time ?v - vehicle)                ; 车辆v的行驶时间
        (maintenance-cost ?v - vehicle)           ; 车辆v的维护成本
    )
    
    (:durative-action drive
        :parameters (?v - vehicle ?r - road ?l1 ?l2 - location)
        :duration (= ?duration (/ (distance ?r) 5.0)) ; 行驶时间 = 距离/速度
        :condition (and 
            (at start (at ?v ?l1))
            (at start (connected ?r ?l1 ?l2))
            (at start (>= (fuel-level ?v) (* (distance ?r) 0.2))) ; 检查燃料
            (over all (has-fuel ?v)))
        :effect (and 
            (at end (not (at ?v ?l1)))
            (at end (at ?v ?l2))
            (at end (decrease (fuel-level ?v) (* (distance ?r) 0.2))) ; 消耗燃料
            (at end (increase (travel-time ?v) ?duration))
            (at end (increase (total-time) ?duration))
            (at end (increase (maintenance-cost ?v) 
                           (if (road-condition ?r good) 1.0 3.0))))) ; 维护成本
    
    (:durative-action load
        :parameters (?v - vehicle ?p - package ?l - location)
        :duration (= ?duration 1.5) ; 装载时间
        :condition (and 
            (at start (at ?v ?l))
            (at start (package-at ?p ?l))
            (at start (not (loaded ?v ?p)))
            (over all (at ?v ?l)))
        :effect (and 
            (at end (not (package-at ?p ?l)))
            (at end (loaded ?v ?p))
            (at end (increase (total-time) ?duration))
            (at end (increase (maintenance-cost ?v) 0.5)))) ; 装载维护成本
    
    (:durative-action unload
        :parameters (?v - vehicle ?p - package ?l - location)
        :duration (= ?duration 1.0) ; 卸载时间
        :condition (and 
            (at start (at ?v ?l))
            (at start (loaded ?v ?p))
            (at start (destination ?p ?l))
            (over all (at ?v ?l)))
        :effect (and 
            (at end (not (loaded ?v ?p)))
            (at end (package-at ?p ?l))
            (at end (increase (delivery-delay ?p) (- (total-time) ?duration))) ; 计算延迟
            (at end (increase (total-time) ?duration))
            (at end (increase (maintenance-cost ?v) 0.3)))) ; 卸载维护成本
    
    (:durative-action refuel
        :parameters (?v - vehicle ?l - location)
        :duration (= ?duration 3.0) ; 加油时间
        :condition (and 
            (at start (at ?v ?l))
            (at start (< (fuel-level ?v) 20))) ; 低燃料触发
        :effect (and 
            (at end (assign (fuel-level ?v) 100))
            (at end (has-fuel ?v))
            (at end (increase (total-time) ?duration))
            (at end (increase (maintenance-cost ?v) 2.0)))) ; 加油维护成本
)    