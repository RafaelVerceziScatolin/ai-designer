(defun C:GetData ( / ss id total ent entData entType save start_x start_y end_x end_y database)
  (setq ss (ssget))
  (if ss
    (progn
      (setq id 0
            total (sslength ss)
            database '()
      )
      (while (< id total)
        (setq ent (ssname ss id)
              entData (entget ent)
              entType (cdr (assoc 0 entData))
        )
        (if (= entType "LINE")
          (progn
            (setq entHandle (cdr (assoc 5 entData))
                  start_x (car (cdr (assoc 10 entData)))
                  start_y (cadr (cdr (assoc 10 entData)))
                  end_x (car (cdr (assoc 11 entData)))
                  end_y (cadr (cdr (assoc 11 entData)))
            )
            (setq save (list entHandle start_x start_y end_x end_y))
            (setq database (append database (list save)))
          )
        )
        (setq id (1+ id))
      )
      
      ; Create the CSV file
      (setq file_path "C:\\Users\\Rafael\\Desktop\\text_recognition\\database.csv")
      (setq file (open file_path "w"))
      (foreach row database
        (setq line (apply 'strcat (mapcar '(lambda (x) (strcat (vl-princ-to-string x) ",")) row)))
        (setq line (substr line 1 (- (strlen line) 1))) ; Remove trailing comma
        (write-line line file)
      )
      (close file)
      
      ;(startapp "cmd.exe" (strcat "/C C:\\Users\\Rafael\\anaconda3\\envs\\dl_torch\\python.exe " "C:\\Users\\Rafael\\Desktop\\autocad_ml\\model_inference.py " file_path))
      
      ;; 5) Wait until Python creates temp.txt
      ;(while (not (findfile "C:\\Users\\Rafael\\Desktop\\codes\\temp.txt"))
      ;  (command "DELAY" "1000")  ;; check every 100 ms
      ;)
      ;(command "DELAY" "15000")
      
      ;; 6) Read the result from temp.txt
      ;(setq f (open "C:\\Users\\Rafael\\Desktop\\autocad_ml\\temp.txt" "r"))
      ;(setq predictions (read (read-line f)))
      ;(close f)
      
      ;; 7) (Optional) Delete temp file once you have the result
      ;(vl-file-delete "C:\\Users\\Rafael\\Desktop\\autocad_ml\\temp.txt")
      ;(vl-file-delete "C:\\Users\\Rafael\\Desktop\\autocad_ml\\database.csv")
      
      ;; 8) Print result in the command line
      ;(princ predictions)
      
      ;; 9) Return the result so other LISP code can use it
      (princ)
    )
  )
)

(defun c:ClassifyLines ()
  (foreach pred predictions
    (setq handle (car pred))
    (setq class (cadr pred))

    ;; Map class number to layer name
    (setq layer
      (cond
        ((= class 0) "beam")
        ((= class 1) "column")
        ((= class 2) "eave")
        ((= class 3) "slab_hole")
        ((= class 4) "stair")
        ((= class 5) "section")
        ((= class 6) "info")
        (T "unknown") ; fallback
      )
    )

    ;; Get entity by handle
    (setq ent (handent handle))

    ;; If entity exists, change its layer
    (if ent
      (progn
        (setq entData (entget ent))
        (setq entData (subst (cons 8 layer) (assoc 8 entData) entData))
        (entmod entData)
      )
    )
  )

  (princ "\nEntities successfully classified into layers.")
  (princ)
)