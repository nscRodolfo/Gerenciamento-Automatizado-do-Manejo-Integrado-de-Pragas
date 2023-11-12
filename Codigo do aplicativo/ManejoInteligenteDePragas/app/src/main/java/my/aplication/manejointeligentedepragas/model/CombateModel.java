package my.aplication.manejointeligentedepragas.model;

public class CombateModel {

    private int fk_Cod_Praga;
    private int fk_Cod_Inimigo_Natural;

    public CombateModel(int fk_Cod_Praga, int fk_Cod_Inimigo_Natural) {
        this.fk_Cod_Praga = fk_Cod_Praga;
        this.fk_Cod_Inimigo_Natural = fk_Cod_Inimigo_Natural;
    }

    public int getFk_Cod_Praga() {
        return fk_Cod_Praga;
    }

    public void setFk_Cod_Praga(int fk_Cod_Praga) {
        this.fk_Cod_Praga = fk_Cod_Praga;
    }

    public int getFk_Cod_Inimigo_Natural() {
        return fk_Cod_Inimigo_Natural;
    }

    public void setFk_Cod_Inimigo_Natural(int fk_Cod_Inimigo_Natural) {
        this.fk_Cod_Inimigo_Natural = fk_Cod_Inimigo_Natural;
    }
}
