def creditApproval(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit


def mutatedCreditApproval1(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5: ##################
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval2(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 4 or region == 5: ###############################
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval3(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age > 18: #############################
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval4(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 25: ##############################################
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval5(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 1: ################################
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval6(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 1: ##################################
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval7(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3: ##############################
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval8(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 1 or region == 2: #############################
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval9(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 5000 ##############################
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit


def mutatedCreditApproval10(citizenship, state, age, sex, region, income_class, dependents_number, marital_status):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 2 * income_class #######################
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit
