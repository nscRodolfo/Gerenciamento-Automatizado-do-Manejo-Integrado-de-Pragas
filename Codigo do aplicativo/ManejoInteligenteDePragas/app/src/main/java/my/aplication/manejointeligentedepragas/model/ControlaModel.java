package my.aplication.manejointeligentedepragas.model;

public class ControlaModel {
    private int fk_Cod_MetodoControle;
    private int fk_Cod_Praga;

    public ControlaModel(int fk_Cod_MetodoControle, int fk_Cod_Praga) {
        this.fk_Cod_MetodoControle = fk_Cod_MetodoControle;
        this.fk_Cod_Praga = fk_Cod_Praga;
    }

    public int getFk_Cod_MetodoControle() {
        return fk_Cod_MetodoControle;
    }

    public void setFk_Cod_MetodoControle(int fk_Cod_MetodoControle) {
        this.fk_Cod_MetodoControle = fk_Cod_MetodoControle;
    }

    public int getFk_Cod_Praga() {
        return fk_Cod_Praga;
    }

    public void setFk_Cod_Praga(int fk_Cod_Praga) {
        this.fk_Cod_Praga = fk_Cod_Praga;
    }
}
